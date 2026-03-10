from __future__ import annotations

import argparse
import json
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

    def __init__(self, model_id: str = EMBEDDING_MODEL_ID):
        self.model_id = model_id
        self.model_dir = Path(snapshot_download(model_id))
        self.model = CLIPForMultiModalEmbedding(str(self.model_dir))
        self.transform = self._build_transform()

    def _build_transform(self) -> Compose:
        with (self.model_dir / "vision_model_config.json").open(encoding="utf-8") as file:
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

    def embed_images(
        self,
        image_paths: list[Path],
        batch_size: int,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> np.ndarray:
        batches: list[np.ndarray] = []
        total_count = len(image_paths)

        emit_progress(
            progress_callback,
            stage="embedding",
            current=0,
            total=total_count,
            message="开始计算图片 embedding",
            model_id=self.model_id,
        )

        with torch.no_grad():
            for start_index in range(0, total_count, batch_size):
                ensure_not_cancelled(cancel_check)
                batch_paths = image_paths[start_index:start_index + batch_size]
                image_tensors = []
                for image_path in batch_paths:
                    ensure_not_cancelled(cancel_check)
                    with Image.open(image_path) as image:
                        image_tensors.append(self.transform(image))

                batch_tensor = torch.stack(image_tensors, dim=0)
                result = self.model({"img": batch_tensor})
                batch_embeddings = result[OutputKeys.IMG_EMBEDDING].cpu().numpy()
                batches.append(batch_embeddings)
                end_index = start_index + len(batch_paths)
                emit_progress(
                    progress_callback,
                    stage="embedding",
                    current=end_index,
                    total=total_count,
                    message=f"已完成 {end_index}/{total_count} 张图片 embedding",
                    model_id=self.model_id,
                )

        return np.vstack(batches) if batches else np.empty((0, 512), dtype=np.float32)


def build_cached_jpg_path(source_path: Path, output_dir: Path) -> Path:
    parent_name = source_path.parent.name or "root"
    return output_dir / f"{parent_name}__{source_path.stem}.jpg"


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

    clusters = []
    singletons = []
    for cluster_id, indices in sorted(
        cluster_map.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        cluster_entry = {
            "cluster_id": cluster_id,
            "size": len(indices),
            "average_similarity": compute_cluster_average_similarity(
                indices, similarity_matrix
            ),
            "items": [
                {
                    "source_path": str(items[index].source_path),
                    "jpg_path": str(items[index].jpg_path),
                }
                for index in indices
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
    cluster_similarity_threshold: float = 0.88,
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
    embedder = ModelScopeClipImageEmbedder()
    emit_progress(
        progress_callback,
        stage="model",
        current=1,
        total=1,
        message="ModelScope embedding 模型加载完成",
        model_id=embedder.model_id,
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
        default=0.88,
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