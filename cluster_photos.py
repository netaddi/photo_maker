from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.multi_modal.clip.model import CLIPForMultiModalEmbedding
from modelscope.outputs import OutputKeys

from convert_raw_to_jpg import OUTPUT_DIR, SUPPORTED_EXTENSIONS, convert_raw_to_jpg


EMBEDDING_MODEL_ID = "damo/multi-modal_clip-vit-base-patch16_zh"
EMBEDDING_CACHE_DIRNAME = ".embedding_cache"
DIRECT_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
CONVERTIBLE_IMAGE_EXTENSIONS = {".png", ".webp"}
ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class PhotoItem:
    source_path: Path
    jpg_path: Path


class JobCancelledError(RuntimeError):
    pass


@dataclass
class LoadedEmbeddingModel:
    model_dir: Path
    model: CLIPForMultiModalEmbedding
    transform: Compose
    inference_lock: threading.Lock


def emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
    **extra: Any,
) -> None:
    if progress_callback is None:
        return

    payload: dict[str, Any] = {
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
        "percent": 100.0 if total == 0 else current / total * 100,
    }
    payload.update(extra)
    progress_callback(payload)


def ensure_not_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise JobCancelledError("任务已取消")


class ModelScopeClipImageEmbedder:

    _loaded_models: dict[str, LoadedEmbeddingModel] = {}
    _loaded_models_lock = threading.Lock()

    def __init__(self, model_id: str = EMBEDDING_MODEL_ID, cache_root: Path = OUTPUT_DIR):
        self.model_id = model_id
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_dir = self.cache_root / EMBEDDING_CACHE_DIRNAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        loaded_model, self.reused_model = self._get_loaded_model(model_id)
        self.model_dir = loaded_model.model_dir
        self.model = loaded_model.model
        self.transform = loaded_model.transform
        self.inference_lock = loaded_model.inference_lock

    @staticmethod
    def _build_transform(model_dir: Path) -> Compose:
        with (model_dir / "vision_model_config.json").open(encoding="utf-8") as file:
            image_resolution = json.load(file)["image_resolution"]

        return Compose(
            [
                Resize((image_resolution, image_resolution), interpolation=Image.BICUBIC),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @classmethod
    def _get_loaded_model(cls, model_id: str) -> tuple[LoadedEmbeddingModel, bool]:
        with cls._loaded_models_lock:
            cached_model = cls._loaded_models.get(model_id)
            if cached_model is not None:
                return cached_model, True

            model_dir = Path(snapshot_download(model_id))
            loaded_model = LoadedEmbeddingModel(
                model_dir=model_dir,
                model=CLIPForMultiModalEmbedding(str(model_dir)),
                transform=cls._build_transform(model_dir),
                inference_lock=threading.Lock(),
            )
            cls._loaded_models[model_id] = loaded_model
            return loaded_model, False

    def _embedding_cache_path(self, image_path: Path) -> Path:
        cache_key = hashlib.sha256(
            f"{self.model_id}\n{image_path}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.npz"

    def _load_cached_embedding(self, image_path: Path) -> np.ndarray | None:
        cache_path = self._embedding_cache_path(image_path)
        if not cache_path.exists():
            return None

        stat = image_path.stat()
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_model_id = str(cached["model_id"].item())
                cached_path = str(cached["image_path"].item())
                cached_mtime_ns = int(cached["mtime_ns"].item())
                cached_size = int(cached["file_size"].item())
                embedding = np.array(cached["embedding"], dtype=np.float32)
        except Exception:
            return None

        if cached_model_id != self.model_id:
            return None
        if cached_path != str(image_path):
            return None
        if cached_mtime_ns != stat.st_mtime_ns or cached_size != stat.st_size:
            return None
        return embedding

    def _save_cached_embedding(self, image_path: Path, embedding: np.ndarray) -> None:
        cache_path = self._embedding_cache_path(image_path)
        stat = image_path.stat()
        np.savez_compressed(
            cache_path,
            model_id=np.array(self.model_id),
            image_path=np.array(str(image_path)),
            mtime_ns=np.array(stat.st_mtime_ns, dtype=np.int64),
            file_size=np.array(stat.st_size, dtype=np.int64),
            embedding=np.asarray(embedding, dtype=np.float32),
        )

    def embed_images(
        self,
        image_paths: list[Path],
        batch_size: int,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> np.ndarray:
        total_count = len(image_paths)
        embeddings_by_index: list[np.ndarray | None] = [None] * total_count
        uncached_indexes: list[int] = []
        uncached_paths: list[Path] = []
        done_count = 0
        cache_hit_count = 0

        emit_progress(
            progress_callback,
            stage="embedding",
            current=0,
            total=total_count,
            message="开始计算图片 embedding",
            model_id=self.model_id,
        )

        for index, image_path in enumerate(image_paths):
            ensure_not_cancelled(cancel_check)
            cached_embedding = self._load_cached_embedding(image_path)
            if cached_embedding is not None:
                embeddings_by_index[index] = cached_embedding
                done_count += 1
                cache_hit_count += 1
                emit_progress(
                    progress_callback,
                    stage="embedding",
                    current=done_count,
                    total=total_count,
                    message=(
                        f"已完成 {done_count}/{total_count} 张图片 embedding "
                        f"(缓存命中 {cache_hit_count})"
                    ),
                    model_id=self.model_id,
                    cache_hits=cache_hit_count,
                )
            else:
                uncached_indexes.append(index)
                uncached_paths.append(image_path)

        with torch.no_grad():
            for start_index in range(0, len(uncached_paths), batch_size):
                ensure_not_cancelled(cancel_check)
                batch_paths = uncached_paths[start_index:start_index + batch_size]
                batch_indexes = uncached_indexes[start_index:start_index + batch_size]
                image_tensors = []
                for image_path in batch_paths:
                    ensure_not_cancelled(cancel_check)
                    with Image.open(image_path) as image:
                        image_tensors.append(self.transform(image))

                batch_tensor = torch.stack(image_tensors, dim=0)
                with self.inference_lock:
                    result = self.model({"img": batch_tensor})
                batch_embeddings = result[OutputKeys.IMG_EMBEDDING].cpu().numpy()

                for batch_offset, image_path in enumerate(batch_paths):
                    embedding = np.asarray(batch_embeddings[batch_offset], dtype=np.float32)
                    embeddings_by_index[batch_indexes[batch_offset]] = embedding
                    self._save_cached_embedding(image_path, embedding)

                done_count += len(batch_paths)
                emit_progress(
                    progress_callback,
                    stage="embedding",
                    current=done_count,
                    total=total_count,
                    message=(
                        f"已完成 {done_count}/{total_count} 张图片 embedding "
                        f"(缓存命中 {cache_hit_count})"
                    ),
                    model_id=self.model_id,
                    cache_hits=cache_hit_count,
                )

        if not embeddings_by_index:
            return np.empty((0, 512), dtype=np.float32)

        return np.vstack([embedding for embedding in embeddings_by_index if embedding is not None])


def build_cached_jpg_path(source_path: Path, output_dir: Path) -> Path:
    parent_name = source_path.parent.name or "root"
    return output_dir / f"{parent_name}__{source_path.stem}.jpg"


def embedding_cache_dir(output_dir: Path) -> Path:
    return output_dir.expanduser().resolve() / EMBEDDING_CACHE_DIRNAME


def get_embedding_cache_status(output_dir: Path) -> dict[str, object]:
    cache_dir = embedding_cache_dir(output_dir)
    file_count = 0
    total_bytes = 0

    if cache_dir.exists():
        for path in cache_dir.rglob("*"):
            if path.is_file():
                file_count += 1
                total_bytes += path.stat().st_size

    return {
        "cache_dir": str(cache_dir),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "exists": cache_dir.exists(),
    }


def clear_embedding_cache(output_dir: Path) -> tuple[Path, int]:
    cache_dir = embedding_cache_dir(output_dir)
    removed_files = 0
    if cache_dir.exists():
        removed_files = sum(1 for path in cache_dir.rglob("*") if path.is_file())
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, removed_files


def default_cluster_result_path(input_dir: Path, output_dir: Path) -> Path:
    directory_name = input_dir.name or "root"
    return output_dir / f"{directory_name}__clusters.json"


def convert_standard_image_to_jpg(source_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jpg_path = build_cached_jpg_path(source_path, output_dir)
    if jpg_path.exists():
        return jpg_path

    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        rgb_image.save(jpg_path, format="JPEG", quality=95, subsampling=0)

    return jpg_path


def iter_photo_files(input_dir: Path) -> list[Path]:
    supported_extensions = (
        SUPPORTED_EXTENSIONS | DIRECT_IMAGE_EXTENSIONS | CONVERTIBLE_IMAGE_EXTENSIONS
    )
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    )


def prepare_photo_items(
    input_dir: Path,
    output_dir: Path,
    limit: int | None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    workers: int = 8,
) -> list[PhotoItem]:
    source_paths = iter_photo_files(input_dir)
    if limit is not None:
        source_paths = source_paths[:limit]

    total_count = len(source_paths)
    emit_progress(
        progress_callback,
        stage="prepare",
        current=0,
        total=total_count,
        message="开始准备 JPG 缓存",
    )

    if total_count == 0:
        return []

    items_by_index: list[PhotoItem | None] = [None] * total_count

    def prepare_one(index_and_path: tuple[int, Path]) -> tuple[int, PhotoItem]:
        index, source_path = index_and_path
        ensure_not_cancelled(cancel_check)
        suffix = source_path.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            jpg_path, _ = convert_raw_to_jpg(source_path, output_dir)
        elif suffix in DIRECT_IMAGE_EXTENSIONS:
            jpg_path = source_path
        else:
            jpg_path = convert_standard_image_to_jpg(source_path, output_dir)

        return index, PhotoItem(source_path=source_path, jpg_path=jpg_path)

    source_items = list(enumerate(source_paths))
    completed_count = 0
    future_map: dict[Future[tuple[int, PhotoItem]], tuple[int, Path]] = {}
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        next_submit_index = 0
        while next_submit_index < len(source_items) and len(future_map) < workers:
            index_and_path = source_items[next_submit_index]
            future = executor.submit(prepare_one, index_and_path)
            future_map[future] = index_and_path
            next_submit_index += 1

        while future_map:
            ensure_not_cancelled(cancel_check)
            done_futures, _ = wait(future_map.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                _, source_path = future_map.pop(future)
                index, item = future.result()
                items_by_index[index] = item
                completed_count += 1
                emit_progress(
                    progress_callback,
                    stage="prepare",
                    current=completed_count,
                    total=total_count,
                    message=f"已准备 {completed_count}/{total_count} 张图片",
                    source_path=str(source_path),
                    jpg_path=str(item.jpg_path),
                )

                if next_submit_index < len(source_items):
                    next_item = source_items[next_submit_index]
                    next_future = executor.submit(prepare_one, next_item)
                    future_map[next_future] = next_item
                    next_submit_index += 1
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    return [item for item in items_by_index if item is not None]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0,), dtype=int)
    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return clusterer.fit_predict(embeddings)


def compute_cluster_average_similarity(
    indices: list[int], similarity_matrix: np.ndarray
) -> float:
    if len(indices) < 2:
        return 1.0

    similarities: list[float] = []
    for left_index, source_index in enumerate(indices):
        for target_index in indices[left_index + 1:]:
            similarities.append(float(similarity_matrix[source_index, target_index]))

    return sum(similarities) / len(similarities)


def photo_item_sort_key(item: PhotoItem) -> tuple[int, str, str]:
    stat = item.source_path.stat()
    return (
        stat.st_mtime_ns,
        item.source_path.name.lower(),
        str(item.source_path),
    )


def build_result(
    items: list[PhotoItem],
    embeddings: np.ndarray,
    labels: np.ndarray,
    similarity_matrix: np.ndarray,
    model_id: str,
    input_dir: Path,
    output_dir: Path,
    cluster_similarity_threshold: float,
    cluster_distance_threshold: float,
    result_path: Path,
) -> dict[str, object]:
    cluster_map: dict[int, list[int]] = {}
    for item_index, label in enumerate(labels.tolist()):
        cluster_map.setdefault(int(label), []).append(item_index)

    sorted_cluster_entries: list[tuple[int, list[int]]] = []
    for cluster_id, indices in cluster_map.items():
        sorted_indices = sorted(indices, key=lambda index: photo_item_sort_key(items[index]))
        sorted_cluster_entries.append((cluster_id, sorted_indices))

    sorted_cluster_entries.sort(
        key=lambda entry: (
            -len(entry[1]),
            photo_item_sort_key(items[entry[1][0]]),
            entry[0],
        )
    )

    clusters = []
    singletons = []
    for cluster_id, indices in sorted_cluster_entries:
        cluster_entry = {
            "cluster_id": cluster_id,
            "size": len(indices),
            "average_similarity": compute_cluster_average_similarity(
                indices, similarity_matrix
            ),
            "items": [
                {
                    "image_id": image_id,
                    "source_path": str(items[index].source_path),
                    "jpg_path": str(items[index].jpg_path),
                }
                for image_id, index in enumerate(indices)
            ],
        }
        if len(indices) > 1:
            clusters.append(cluster_entry)
        else:
            singletons.extend(cluster_entry["items"])

    return {
        "input_dir": str(input_dir),
        "cache_output_dir": str(output_dir),
        "result_path": str(result_path),
        "embedding_model_id": model_id,
        "total_photos": len(items),
        "cluster_count": len(clusters),
        "singleton_count": len(singletons),
        "cluster_similarity_threshold": cluster_similarity_threshold,
        "cluster_distance_threshold": cluster_distance_threshold,
        "clusters": clusters,
        "singletons": singletons,
        "embedding_shape": list(embeddings.shape),
    }


def _cluster_identity(cluster: dict[str, object]) -> tuple[str, ...]:
    items = cluster.get("items", [])
    if not isinstance(items, list):
        return ()
    source_paths = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("source_path"), str):
            source_paths.append(item["source_path"])
    return tuple(source_paths)


def preserve_existing_vlm_pick_cache(result: dict[str, object], result_path: Path) -> dict[str, object]:
    if not result_path.exists() or not result_path.is_file():
        return result

    try:
        with result_path.open("r", encoding="utf-8") as file:
            existing_result = json.load(file)
    except Exception:
        return result

    if not isinstance(existing_result, dict):
        return result

    existing_clusters = existing_result.get("clusters", [])
    if not isinstance(existing_clusters, list):
        return result

    existing_cluster_map: dict[tuple[str, ...], dict[str, object]] = {}
    for cluster in existing_clusters:
        if not isinstance(cluster, dict):
            continue
        identity = _cluster_identity(cluster)
        if identity:
            existing_cluster_map[identity] = cluster

    result_clusters = result.get("clusters", [])
    if not isinstance(result_clusters, list):
        return result

    for cluster in result_clusters:
        if not isinstance(cluster, dict):
            continue
        identity = _cluster_identity(cluster)
        existing_cluster = existing_cluster_map.get(identity)
        if existing_cluster is None:
            continue
        existing_pick = existing_cluster.get("vlm_pick")
        if isinstance(existing_pick, dict):
            cluster["vlm_pick"] = existing_pick

    existing_top_level_pick = existing_result.get("vlm_pick")
    if isinstance(existing_top_level_pick, dict) and "vlm_pick" not in result:
        result["vlm_pick"] = existing_top_level_pick

    return result


def write_result(result: dict[str, object], result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def validate_args(
    input_dir: Path,
    batch_size: int,
    cluster_distance_threshold: float,
    limit: int | None,
    workers: int,
) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")
    if batch_size < 1:
        raise SystemExit("batch-size must be at least 1")
    if not 0.0 <= cluster_distance_threshold <= 2.0:
        raise SystemExit("cluster-distance-threshold must be between 0.0 and 2.0")
    if limit is not None and limit < 1:
        raise SystemExit("limit must be at least 1 when provided")
    if workers < 1:
        raise SystemExit("workers must be at least 1")


def similarity_threshold_to_distance(similarity_threshold: float) -> float:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise SystemExit("cluster-similarity-threshold must be between 0.0 and 1.0")
    return 1.0 - similarity_threshold


def run_clustering(
    input_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    result_path: Path | None = None,
    batch_size: int = 16,
    workers: int = 8,
    cluster_similarity_threshold: float = 0.97,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> tuple[dict[str, object], Path]:
    resolved_input_dir = input_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    cluster_distance_threshold = similarity_threshold_to_distance(
        cluster_similarity_threshold
    )
    validate_args(
        resolved_input_dir,
        batch_size,
        cluster_distance_threshold,
        limit,
        workers,
    )

    resolved_result_path = (
        result_path.expanduser().resolve()
        if result_path is not None
        else default_cluster_result_path(resolved_input_dir, resolved_output_dir)
    )

    emit_progress(
        progress_callback,
        stage="scan",
        current=0,
        total=1,
        message="开始扫描目录",
        input_dir=str(resolved_input_dir),
    )

    photo_items = prepare_photo_items(
        resolved_input_dir,
        resolved_output_dir,
        limit,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        workers=workers,
    )
    if not photo_items:
        raise SystemExit(f"No supported photos found in: {resolved_input_dir}")

    emit_progress(
        progress_callback,
        stage="model",
        current=0,
        total=1,
        message="开始加载 ModelScope embedding 模型",
        model_id=EMBEDDING_MODEL_ID,
    )
    embedder = ModelScopeClipImageEmbedder(cache_root=resolved_output_dir)
    emit_progress(
        progress_callback,
        stage="model",
        current=1,
        total=1,
        message=(
            "复用已加载的 ModelScope embedding 模型"
            if embedder.reused_model
            else "ModelScope embedding 模型加载完成"
        ),
        model_id=embedder.model_id,
        model_reused=embedder.reused_model,
    )

    embeddings = embedder.embed_images(
        [item.jpg_path for item in photo_items],
        batch_size=batch_size,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    ensure_not_cancelled(cancel_check)
    emit_progress(
        progress_callback,
        stage="cluster",
        current=0,
        total=1,
        message="开始计算相似度并聚类",
    )
    labels = cluster_embeddings(embeddings, cluster_distance_threshold)
    similarity_matrix = cosine_similarity_matrix(embeddings)
    ensure_not_cancelled(cancel_check)

    result = build_result(
        items=photo_items,
        embeddings=embeddings,
        labels=labels,
        similarity_matrix=similarity_matrix,
        model_id=embedder.model_id,
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        cluster_similarity_threshold=cluster_similarity_threshold,
        cluster_distance_threshold=cluster_distance_threshold,
        result_path=resolved_result_path,
    )
    result = preserve_existing_vlm_pick_cache(result, resolved_result_path)

    emit_progress(
        progress_callback,
        stage="write",
        current=0,
        total=1,
        message="开始写入聚类结果 JSON",
        result_path=str(resolved_result_path),
    )
    ensure_not_cancelled(cancel_check)
    write_result(result, resolved_result_path)
    emit_progress(
        progress_callback,
        stage="done",
        current=1,
        total=1,
        message="聚类完成",
        result_path=str(resolved_result_path),
        result=result,
    )
    return result, resolved_result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert photos to JPEG when needed, compute ModelScope image embeddings, and cluster similar photos."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing photos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for cached JPEG files. Defaults to {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="Path to the JSON file that will store clustering results. Defaults to the JPEG cache directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for embedding extraction. Defaults to 16.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads used while preparing cached JPEG files. Defaults to 8.",
    )
    parser.add_argument(
        "--cluster-similarity-threshold",
        type=float,
        default=0.97,
        help="Cluster similarity threshold between 0.0 and 1.0. Higher values produce tighter clusters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of photos to process. Useful for smoke tests.",
    )
    return parser.parse_args()


def print_console_progress(event: dict[str, Any]) -> None:
    stage = event["stage"]
    current = int(event["current"])
    total = int(event["total"])
    message = str(event["message"])
    if stage in {"prepare", "embedding"}:
        print(f"\r{message}", end="", flush=True)
        if current == total:
            print(flush=True)
        return

    print(message, flush=True)


def main() -> int:
    args = parse_args()
    result, result_path = run_clustering(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        result_path=args.result_path,
        batch_size=args.batch_size,
        workers=args.workers,
        cluster_similarity_threshold=args.cluster_similarity_threshold,
        limit=args.limit,
        progress_callback=print_console_progress,
    )

    print(
        "SUMMARY "
        f"photos={result['total_photos']} clusters={result['cluster_count']} "
        f"singletons={result['singleton_count']}"
    )
    print(f"RESULT {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())