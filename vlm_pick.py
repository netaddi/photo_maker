from __future__ import annotations

import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib import error, request

from convert_raw_to_jpg import jpg_file_to_base64


DEFAULT_VLM_PICK_PROMPT = (
    "你是专业摄影编辑。请从同一组相似照片中选出最值得保留的一张。"
    "对于人物照片，你需要优先考虑主体状态、表情/姿态，以及人物面部光线，不可以选择任何表情狰狞、扭曲、闭眼的照片。人物的面部是最重要的。"
    "对于风景照片，你需要优先考虑是否有明显的瑕疵（模糊、噪点等），其次考虑画面主体性和完整性，尤其是是否有噪点。"
    "你可以先分析，但最后必须单独输出一行 JSON，格式严格为 {\"select\": 0}，首张图片为第0张。"
)

SYSTEM_PROMPT = (
    "You are a professional photography editor. Compare similar photos carefully, "
    "think through tradeoffs if needed, and finally choose exactly one image id."
)

VlmEventCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class VlmPickCancelledError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint is required")
    return normalized


def ensure_not_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise VlmPickCancelledError("VLM pick cancelled")


def api_url(endpoint: str, suffix: str) -> str:
    return f"{normalize_endpoint(endpoint)}{suffix}"


def build_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_json_request(
    method: str,
    url: str,
    *,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, method=method, headers=build_headers(api_key), data=data)
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def fetch_model_ids(endpoint: str, api_key: str = "") -> list[str]:
    payload = http_json_request("GET", api_url(endpoint, "/models"), api_key=api_key)
    data = payload.get("data", [])
    models = sorted(
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    return models


def ensure_cluster_item_ids(result: dict[str, Any]) -> None:
    for cluster in result.get("clusters", []):
        items = cluster.get("items", [])
        for image_id, item in enumerate(items):
            if isinstance(item, dict):
                item["image_id"] = image_id


def build_messages(cluster: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{prompt}\n\n"
                "你会收到同一组相似照片。每张图片都带有从 0 开始的 image_id。\n"
                "请比较所有图片，可以先分析，再给出结论。\n"
                "最后一行必须只输出 JSON，格式为 {\"select\": <整数>}。\n"
                "不要输出 markdown 代码块。"
            ),
        }
    ]

    for item in cluster.get("items", []):
        image_id = item.get("image_id")
        jpg_path = item.get("jpg_path")
        if not isinstance(image_id, int) or not isinstance(jpg_path, str):
            continue
        content.append({
            "type": "text",
            "text": f"图片 image_id={image_id}。请记住这个编号。",
        })
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{jpg_file_to_base64(Path(jpg_path))}"
                },
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def iter_stream_json_chunks(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    cancel_check: CancelCheck | None = None,
) -> Iterator[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    req = request.Request(
        api_url(endpoint, "/chat/completions"),
        method="POST",
        headers=build_headers(api_key),
        data=json.dumps(payload).encode("utf-8"),
    )
    ensure_not_cancelled(cancel_check)
    try:
        with request.urlopen(req, timeout=5) as response:
            while True:
                ensure_not_cancelled(cancel_check)
                try:
                    raw_line = response.readline()
                except socket.timeout:
                    ensure_not_cancelled(cancel_check)
                    continue
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                yield json.loads(payload_text)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Streaming request failed: {exc}") from exc


def iter_delta_texts(chunk: dict[str, Any]) -> Iterator[tuple[str, bool]]:
    choices = chunk.get("choices", [])
    if not choices:
        return
    delta = choices[0].get("delta", {})

    for key in ("reasoning_content", "thinking", "reasoning"):
        value = delta.get(key)
        for text in normalize_delta_value(value):
            yield text, True

    for text in normalize_delta_value(delta.get("content")):
        yield text, False


def normalize_delta_value(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    yield text


def extract_selection_json(text: str) -> dict[str, Any] | None:
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "select" in payload:
            return payload
    return None


def parse_selected_image_id(text: str, cluster_size: int) -> tuple[int | None, dict[str, Any] | None]:
    parsed_json = extract_selection_json(text)
    if parsed_json is not None:
        value = parsed_json.get("select")
        if isinstance(value, int) and 0 <= value < cluster_size:
            return value, parsed_json

    patterns = [
        r'"select"\s*:\s*(-?\d+)',
        r"'select'\s*:\s*(-?\d+)",
        r"select\s*[:=]\s*(-?\d+)",
        r'"selected_image_id"\s*:\s*(-?\d+)',
        r"'selected_image_id'\s*:\s*(-?\d+)",
        r"selected_image_id\s*[:=]\s*(-?\d+)",
        r"选择图片id\s*[:：]?\s*(-?\d+)",
        r"选择的图片id\s*[:：]?\s*(-?\d+)",
        r"^\s*(-?\d+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        value = int(match.group(1))
        if 0 <= value < cluster_size:
            return value, {"select": value}
    return None, parsed_json


def load_result_json(result_path: Path) -> dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Cluster result JSON must be an object")
    ensure_cluster_item_ids(payload)
    return payload


def write_result_json(result_path: Path, payload: dict[str, Any]) -> None:
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def model_cache_key(endpoint: str, model: str) -> str:
    return f"{normalize_endpoint(endpoint)}|{model.strip()}"


def model_cache_label(model_entry: dict[str, Any]) -> str:
    model = str(model_entry.get("model") or "").strip()
    endpoint = str(model_entry.get("endpoint") or "").strip()
    if not endpoint:
        return model
    host = endpoint.replace("http://", "").replace("https://", "").rstrip("/")
    return f"{model} @ {host}"


def normalize_cluster_vlm_pick(cluster: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    existing_pick = cluster.get("vlm_pick")
    if not isinstance(existing_pick, dict):
        return None

    by_model = existing_pick.get("by_model")
    if not isinstance(by_model, dict):
        by_model = {}

    if by_model:
        existing_pick["by_model"] = by_model
        return existing_pick

    legacy_endpoint = str(existing_pick.get("endpoint") or payload.get("vlm_pick", {}).get("endpoint") or "").strip()
    legacy_model = str(existing_pick.get("model") or payload.get("vlm_pick", {}).get("model") or "").strip()
    legacy_select = existing_pick.get("select")
    if not isinstance(legacy_select, int):
        legacy_select = existing_pick.get("selected_image_id")

    if legacy_endpoint and legacy_model and isinstance(legacy_select, int):
        entry = dict(existing_pick)
        entry["select"] = legacy_select
        entry["selected_image_id"] = legacy_select
        entry["model_key"] = model_cache_key(legacy_endpoint, legacy_model)
        entry["label"] = model_cache_label(entry)
        by_model[entry["model_key"]] = entry
        existing_pick["by_model"] = by_model
        existing_pick["current_model_key"] = entry["model_key"]
        existing_pick["select"] = legacy_select
        existing_pick["selected_image_id"] = legacy_select
        existing_pick["label"] = entry["label"]
        cluster["vlm_pick"] = existing_pick
        return existing_pick

    return existing_pick


def clear_vlm_pick_results(result_path: Path) -> dict[str, Any]:
    payload = load_result_json(result_path)
    cleared_clusters = 0
    for cluster in payload.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        if "vlm_pick" in cluster:
            cleared_clusters += 1
            cluster.pop("vlm_pick", None)

    had_top_level_pick = isinstance(payload.get("vlm_pick"), dict)
    payload.pop("vlm_pick", None)
    write_result_json(result_path, payload)
    return {
        "result_path": str(result_path),
        "cleared_clusters": cleared_clusters,
        "had_top_level_pick": had_top_level_pick,
        "result": payload,
    }


def run_vlm_pick(
    *,
    result_path: Path,
    endpoint: str,
    model: str,
    prompt: str,
    api_key: str = "",
    only_unpicked: bool = True,
    overwrite_existing: bool = False,
    event_callback: VlmEventCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    normalized_endpoint = normalize_endpoint(endpoint)
    if not model.strip():
        raise ValueError("model is required")
    if not prompt.strip():
        raise ValueError("prompt is required")

    ensure_not_cancelled(cancel_check)

    payload = load_result_json(result_path)
    clusters = payload.get("clusters", [])
    if not isinstance(clusters, list):
        raise ValueError("Invalid cluster result JSON: clusters must be a list")

    payload["vlm_pick"] = {
        "endpoint": normalized_endpoint,
        "model": model,
        "model_key": model_cache_key(normalized_endpoint, model),
        "prompt": prompt,
        "selection_format": {"select": 0},
        "only_unpicked": only_unpicked,
        "overwrite_existing": overwrite_existing,
        "updated_at": utc_now(),
        "cluster_count": len(clusters),
    }
    current_model_key = str(payload["vlm_pick"]["model_key"])

    for cluster_index, cluster in enumerate(clusters):
        ensure_not_cancelled(cancel_check)
        if not isinstance(cluster, dict):
            continue

        items = cluster.get("items", [])
        if not isinstance(items, list) or not items:
            continue

        existing_pick = normalize_cluster_vlm_pick(cluster, payload)
        existing_selected_value = None
        existing_model_pick = None
        if isinstance(existing_pick, dict):
            by_model = existing_pick.get("by_model")
            if isinstance(by_model, dict):
                model_entry = by_model.get(current_model_key)
                if isinstance(model_entry, dict):
                    existing_model_pick = model_entry
                    if isinstance(model_entry.get("select"), int):
                        existing_selected_value = model_entry.get("select")
                    elif isinstance(model_entry.get("selected_image_id"), int):
                        existing_selected_value = model_entry.get("selected_image_id")
            elif isinstance(existing_pick.get("select"), int):
                existing_selected_value = existing_pick.get("select")
                existing_model_pick = existing_pick
            elif isinstance(existing_pick.get("selected_image_id"), int):
                existing_selected_value = existing_pick.get("selected_image_id")
                existing_model_pick = existing_pick

        has_existing_pick = isinstance(existing_selected_value, int)
        if overwrite_existing:
            should_process = True
        elif only_unpicked:
            should_process = not has_existing_pick
        else:
            should_process = True

        if not should_process:
            if isinstance(existing_pick, dict) and isinstance(existing_model_pick, dict) and existing_selected_value is not None:
                existing_model_pick.setdefault("select", existing_selected_value)
                existing_model_pick.setdefault("selected_image_id", existing_selected_value)
                existing_model_pick.setdefault("selection_json", {"select": existing_selected_value})
                existing_model_pick.setdefault("model_key", current_model_key)
                existing_model_pick.setdefault("label", model_cache_label(existing_model_pick))
                by_model = existing_pick.setdefault("by_model", {})
                if isinstance(by_model, dict):
                    by_model[current_model_key] = existing_model_pick
                existing_pick["current_model_key"] = current_model_key
                existing_pick["select"] = existing_selected_value
                existing_pick["selected_image_id"] = existing_selected_value
                existing_pick["selection_json"] = existing_model_pick.get("selection_json", {"select": existing_selected_value})
                existing_pick["endpoint"] = normalized_endpoint
                existing_pick["model"] = model
                existing_pick["label"] = existing_model_pick.get("label", model_cache_label(existing_model_pick))
                write_result_json(result_path, payload)

            if event_callback is not None:
                event_callback(
                    {
                        "type": "cluster_skipped",
                        "cluster_index": cluster_index,
                        "cluster_id": cluster.get("cluster_id"),
                        "reason": "already_picked",
                        "select": existing_selected_value,
                    }
                )
            continue

        if event_callback is not None:
            event_callback(
                {
                    "type": "cluster_start",
                    "cluster_index": cluster_index,
                    "cluster_id": cluster.get("cluster_id"),
                    "image_count": len(items),
                }
            )

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        all_parts: list[str] = []
        messages = build_messages(cluster, prompt)

        for chunk in iter_stream_json_chunks(
            normalized_endpoint,
            model,
            messages,
            api_key=api_key,
            cancel_check=cancel_check,
        ):
            ensure_not_cancelled(cancel_check)
            for text, is_thinking in iter_delta_texts(chunk):
                if is_thinking:
                    reasoning_parts.append(text)
                else:
                    content_parts.append(text)
                all_parts.append(text)
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "token",
                            "cluster_index": cluster_index,
                            "cluster_id": cluster.get("cluster_id"),
                            "text": text,
                            "is_thinking": is_thinking,
                        }
                    )

        ensure_not_cancelled(cancel_check)

        content_text = "".join(content_parts).strip()
        transcript_text = "".join(all_parts).strip()
        selected_image_id, selection_json = parse_selected_image_id(content_text, len(items))
        if selected_image_id is None:
            selected_image_id, selection_json = parse_selected_image_id(transcript_text, len(items))

        if selection_json is None and selected_image_id is not None:
            selection_json = {"select": selected_image_id}

        model_entry = {
            "model_key": current_model_key,
            "label": model_cache_label({"model": model, "endpoint": normalized_endpoint}),
            "select": selected_image_id,
            "selected_image_id": selected_image_id,
            "selection_json": selection_json,
            "content_text": content_text,
            "thinking_text": "".join(reasoning_parts).strip(),
            "transcript_text": transcript_text,
            "endpoint": normalized_endpoint,
            "model": model,
            "prompt": prompt,
            "only_unpicked": only_unpicked,
            "overwrite_existing": overwrite_existing,
            "updated_at": utc_now(),
        }

        cluster_pick = cluster.get("vlm_pick") if isinstance(cluster.get("vlm_pick"), dict) else {}
        by_model = cluster_pick.get("by_model") if isinstance(cluster_pick.get("by_model"), dict) else {}
        by_model[current_model_key] = model_entry

        cluster["vlm_pick"] = {
            **cluster_pick,
            "current_model_key": current_model_key,
            "by_model": by_model,
            "label": model_entry["label"],
            "select": selected_image_id,
            "selected_image_id": selected_image_id,
            "selection_json": selection_json,
            "content_text": content_text,
            "thinking_text": model_entry["thinking_text"],
            "transcript_text": transcript_text,
            "endpoint": normalized_endpoint,
            "model": model,
            "prompt": prompt,
            "only_unpicked": only_unpicked,
            "overwrite_existing": overwrite_existing,
            "updated_at": utc_now(),
        }

        write_result_json(result_path, payload)

        if event_callback is not None:
            event_callback(
                {
                    "type": "cluster_done",
                    "cluster_index": cluster_index,
                    "cluster_id": cluster.get("cluster_id"),
                    "selected_image_id": selected_image_id,
                    "select": selected_image_id,
                    "label": model_entry["label"],
                    "content_text": content_text,
                    "thinking_text": cluster["vlm_pick"]["thinking_text"],
                }
            )

    payload["vlm_pick"]["updated_at"] = utc_now()
    write_result_json(result_path, payload)

    if event_callback is not None:
        event_callback(
            {
                "type": "done",
                "result_path": str(result_path),
                "result": payload,
            }
        )

    return payload