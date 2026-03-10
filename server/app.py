from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from cluster_photos import JobCancelledError, run_clustering
from convert_raw_to_jpg import OUTPUT_DIR


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_BROWSE_ROOT = Path("/Volumes")
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Photo Maker Cluster Server")
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


class ClusterRequest(BaseModel):
    input_dir: str
    output_dir: str = str(OUTPUT_DIR)
    result_path: str | None = None
    batch_size: int = 16
    workers: int = 8
    cluster_similarity_threshold: float = 0.88
    limit: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    }


@app.post("/api/pick-directory")
def pick_directory() -> dict[str, str]:
    return {"path": pick_directory_with_system_dialog()}


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