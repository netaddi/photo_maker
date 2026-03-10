from __future__ import annotations

import json
import queue
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from cluster_photos import (
    JobCancelledError,
    clear_embedding_cache,
    get_embedding_cache_status,
    run_clustering,
)
from convert_raw_to_jpg import OUTPUT_DIR
from photo_description_config import API_BASE_URL
from vlm_pick import (
    DEFAULT_VLM_CONCURRENCY,
    DEFAULT_VLM_PICK_PROMPT,
    VlmPickCancelledError,
    clear_vlm_pick_results,
    fetch_model_ids,
    run_vlm_pick,
)


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_BROWSE_ROOT = Path("/Volumes")
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Photo Maker Cluster Server")
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
vlm_jobs: dict[str, dict[str, Any]] = {}
vlm_jobs_lock = threading.Lock()


class ClusterRequest(BaseModel):
    input_dir: str
    output_dir: str = str(OUTPUT_DIR)
    result_path: str | None = None
    batch_size: int = 16
    workers: int = 8
    cluster_similarity_threshold: float = 0.97
    limit: int | None = None


class CacheClearRequest(BaseModel):
    output_dir: str = str(OUTPUT_DIR)


class VlmModelsRequest(BaseModel):
    endpoint: str
    api_key: str = ""


class VlmPickRequest(BaseModel):
    result_path: str
    endpoint: str
    model: str
    prompt: str
    api_key: str = ""
    only_unpicked: bool = True
    overwrite_existing: bool = False
    concurrency: int = Field(default=DEFAULT_VLM_CONCURRENCY, ge=1, le=32)


class VlmClearRequest(BaseModel):
    result_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_vlm_event(job_id: str, event: dict[str, Any]) -> None:
    event_type = str(event.get("type", "unknown"))
    cluster_id = event.get("cluster_id")

    if event_type == "token":
        token_text = str(event.get("text", ""))
        token_kind = "thinking" if event.get("is_thinking") else "content"
        print(
            f"[VLM][{job_id}][cluster={cluster_id}][{token_kind}] {token_text}",
            flush=True,
        )
        return

    if event_type == "cluster_start":
        print(
            f"[VLM][{job_id}] cluster_start cluster={cluster_id} image_count={event.get('image_count')}",
            flush=True,
        )
        return

    if event_type == "cluster_done":
        print(
            f"[VLM][{job_id}] cluster_done cluster={cluster_id} select={event.get('select')} label={event.get('label', '')}",
            flush=True,
        )
        return

    if event_type == "cluster_skipped":
        print(
            f"[VLM][{job_id}] cluster_skipped cluster={cluster_id} select={event.get('select')}",
            flush=True,
        )
        return

    print(f"[VLM][{job_id}] {json.dumps(event, ensure_ascii=False)}", flush=True)


def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


def update_job(job_id: str, **changes: Any) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].update(changes)


def is_job_cancelled(job_id: str) -> bool:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return True
        return bool(job.get("cancel_requested", False))


def has_active_jobs() -> bool:
    with jobs_lock:
        has_cluster_jobs = any(job["status"] in {"running", "cancelling"} for job in jobs.values())
    with vlm_jobs_lock:
        has_vlm_jobs = any(job["status"] in {"running", "cancelling"} for job in vlm_jobs.values())
    return has_cluster_jobs or has_vlm_jobs


def get_vlm_job(job_id: str) -> dict[str, Any]:
    with vlm_jobs_lock:
        job = vlm_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="VLM job not found")
        return dict(job)


def update_vlm_job(job_id: str, **changes: Any) -> None:
    with vlm_jobs_lock:
        if job_id not in vlm_jobs:
            return
        vlm_jobs[job_id].update(changes)


def is_vlm_job_cancelled(job_id: str) -> bool:
    with vlm_jobs_lock:
        job = vlm_jobs.get(job_id)
        if job is None:
            return True
        return bool(job.get("cancel_requested", False))


def normalize_directory(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def browse_directory(path_text: str | None) -> dict[str, Any]:
    if path_text:
        directory = normalize_directory(path_text)
    elif DEFAULT_BROWSE_ROOT.exists():
        directory = DEFAULT_BROWSE_ROOT.resolve()
    else:
        directory = Path.home().resolve()

    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    child_directories = sorted(
        child for child in directory.iterdir() if child.is_dir() and not child.name.startswith(".")
    )
    return {
        "current_path": str(directory),
        "parent_path": str(directory.parent) if directory.parent != directory else None,
        "directories": [
            {
                "name": child.name,
                "path": str(child),
            }
            for child in child_directories
        ],
    }


def pick_directory_with_system_dialog() -> str:
    script = 'POSIX path of (choose folder with prompt "选择要聚类的照片目录")'
    completed = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if "User canceled" in stderr or "-128" in stderr:
            raise HTTPException(status_code=409, detail="Directory selection cancelled")
        raise HTTPException(status_code=500, detail=stderr or "Failed to open folder picker")

    return str(Path(completed.stdout.strip()).resolve())


def run_job(job_id: str, request: ClusterRequest) -> None:
    def on_progress(event: dict[str, Any]) -> None:
        update_job(
            job_id,
            stage=event["stage"],
            current=event["current"],
            total=event["total"],
            percent=event["percent"],
            message=event["message"],
            meta={
                key: value
                for key, value in event.items()
                if key not in {"stage", "current", "total", "percent", "message"}
            },
        )

    try:
        result, result_path = run_clustering(
            input_dir=Path(request.input_dir),
            output_dir=Path(request.output_dir),
            result_path=Path(request.result_path) if request.result_path else None,
            batch_size=request.batch_size,
            workers=request.workers,
            cluster_similarity_threshold=request.cluster_similarity_threshold,
            limit=request.limit,
            progress_callback=on_progress,
            cancel_check=lambda: is_job_cancelled(job_id),
        )
    except JobCancelledError:
        update_job(
            job_id,
            status="cancelled",
            stage="cancelled",
            message="任务已取消",
            error=None,
            finished_at=utc_now(),
        )
        return
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            message=str(exc),
            error=str(exc),
            finished_at=utc_now(),
        )
        return

    update_job(
        job_id,
        status="done",
        stage="done",
        current=1,
        total=1,
        percent=100.0,
        message="聚类完成",
        result=result,
        result_path=str(result_path),
        finished_at=utc_now(),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    browse_root = DEFAULT_BROWSE_ROOT if DEFAULT_BROWSE_ROOT.exists() else Path.home()
    return {
        "output_dir": str(OUTPUT_DIR),
        "browse_root": str(browse_root.resolve()),
        "vlm_endpoint": API_BASE_URL,
        "vlm_prompt": DEFAULT_VLM_PICK_PROMPT,
        "vlm_concurrency": DEFAULT_VLM_CONCURRENCY,
    }


@app.post("/api/pick-directory")
def pick_directory() -> dict[str, str]:
    return {"path": pick_directory_with_system_dialog()}


@app.post("/api/cache/embedding/clear")
def clear_embedding_cache_endpoint(request: CacheClearRequest) -> dict[str, Any]:
    if has_active_jobs():
        raise HTTPException(status_code=409, detail="Cannot clear embedding cache while a job is running")

    output_dir = normalize_directory(request.output_dir)
    cache_dir, removed_files = clear_embedding_cache(output_dir)
    return {
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "removed_files": removed_files,
    }


@app.get("/api/cache/embedding/status")
def embedding_cache_status(output_dir: str = Query(default=str(OUTPUT_DIR))) -> dict[str, Any]:
    resolved_output_dir = normalize_directory(output_dir)
    return {
        "output_dir": str(resolved_output_dir),
        **get_embedding_cache_status(resolved_output_dir),
    }


@app.post("/api/vlm/models")
def vlm_models(request: VlmModelsRequest) -> dict[str, Any]:
    try:
        models = fetch_model_ids(request.endpoint, api_key=request.api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "endpoint": request.endpoint,
        "models": models,
    }


@app.post("/api/vlm/picks/clear")
def clear_vlm_picks(request: VlmClearRequest) -> dict[str, Any]:
    if has_active_jobs():
        raise HTTPException(status_code=409, detail="Cannot clear VLM picks while a job is running")

    result_path = Path(request.result_path).expanduser().resolve()
    if not result_path.exists() or not result_path.is_file():
        raise HTTPException(status_code=400, detail=f"Result JSON does not exist: {result_path}")

    try:
        return clear_vlm_pick_results(result_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vlm/pick/stream")
def vlm_pick_stream(request: VlmPickRequest) -> StreamingResponse:
    result_path = Path(request.result_path).expanduser().resolve()
    if not result_path.exists() or not result_path.is_file():
        raise HTTPException(status_code=400, detail=f"Result JSON does not exist: {result_path}")
    if not request.endpoint.strip():
        raise HTTPException(status_code=400, detail="endpoint is required")
    if not request.model.strip():
        raise HTTPException(status_code=400, detail="model is required")
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    vlm_job_id = uuid4().hex
    with vlm_jobs_lock:
        vlm_jobs[vlm_job_id] = {
            "job_id": vlm_job_id,
            "status": "running",
            "message": "VLM 挑图任务已创建，等待开始",
            "result_path": str(result_path),
            "cancel_requested": False,
            "created_at": utc_now(),
            "finished_at": None,
            "error": None,
        }

    def stream() -> Any:
        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            log_vlm_event(vlm_job_id, event)
            event_queue.put(event)

        def worker() -> None:
            try:
                update_vlm_job(vlm_job_id, message="VLM 挑图进行中")
                run_vlm_pick(
                    result_path=result_path,
                    endpoint=request.endpoint,
                    model=request.model,
                    prompt=request.prompt,
                    api_key=request.api_key,
                    only_unpicked=request.only_unpicked,
                    overwrite_existing=request.overwrite_existing,
                    concurrency=request.concurrency,
                    event_callback=emit,
                    cancel_check=lambda: is_vlm_job_cancelled(vlm_job_id),
                )
                update_vlm_job(
                    vlm_job_id,
                    status="done",
                    message="VLM 挑图完成",
                    finished_at=utc_now(),
                )
            except VlmPickCancelledError:
                update_vlm_job(
                    vlm_job_id,
                    status="cancelled",
                    message="VLM 挑图已取消",
                    finished_at=utc_now(),
                )
                print(f"[VLM][{vlm_job_id}] cancelled", flush=True)
                event_queue.put({"type": "cancelled", "job_id": vlm_job_id, "message": "VLM 挑图已取消"})
            except Exception as exc:
                update_vlm_job(
                    vlm_job_id,
                    status="error",
                    message=str(exc),
                    error=str(exc),
                    finished_at=utc_now(),
                )
                print(f"[VLM][{vlm_job_id}] error: {exc}", flush=True)
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        print(
            f"[VLM][{vlm_job_id}] start endpoint={request.endpoint} model={request.model} concurrency={request.concurrency} result_path={result_path}",
            flush=True,
        )
        yield f"data: {json.dumps({'type': 'start', 'job_id': vlm_job_id, 'result_path': str(result_path), 'concurrency': request.concurrency}, ensure_ascii=False)}\n\n"

        while True:
            event = event_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") in {"done", "error", "cancelled"}:
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/vlm/jobs/{job_id}")
def read_vlm_job(job_id: str) -> dict[str, Any]:
    return get_vlm_job(job_id)


@app.post("/api/vlm/jobs/{job_id}/cancel")
def cancel_vlm_job(job_id: str) -> dict[str, str]:
    with vlm_jobs_lock:
        job = vlm_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="VLM job not found")
        if job["status"] in {"done", "error", "cancelled"}:
            return {"status": job["status"]}
        job["cancel_requested"] = True
        job["status"] = "cancelling"
        job["message"] = "正在取消 VLM 挑图，等待当前响应片段结束"
    return {"status": "cancelling"}


@app.get("/api/browse")
def browse(path: str | None = Query(default=None)) -> dict[str, Any]:
    return browse_directory(path)


@app.post("/api/jobs")
def create_job(request: ClusterRequest) -> dict[str, str]:
    input_dir = normalize_directory(request.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Input directory does not exist: {input_dir}")

    output_dir = normalize_directory(request.output_dir)
    request.input_dir = str(input_dir)
    request.output_dir = str(output_dir)
    if request.result_path:
        request.result_path = str(Path(request.result_path).expanduser().resolve())

    job_id = uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "current": 0,
            "total": 1,
            "percent": 0.0,
            "message": "任务已创建，等待开始",
            "input_dir": request.input_dir,
            "output_dir": request.output_dir,
            "result_path": request.result_path,
            "result": None,
            "error": None,
            "meta": {},
            "cancel_requested": False,
            "created_at": utc_now(),
            "finished_at": None,
        }

    worker = threading.Thread(target=run_job, args=(job_id, request), daemon=True)
    worker.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str) -> dict[str, Any]:
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in {"done", "error", "cancelled"}:
            return {"status": job["status"]}
        job["cancel_requested"] = True
        job["status"] = "cancelling"
        job["message"] = "正在取消任务，等待当前批次结束"
    return {"status": "cancelling"}


@app.get("/api/image")
def read_image(path: str = Query(...)) -> FileResponse:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return FileResponse(image_path)