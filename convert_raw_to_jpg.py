from __future__ import annotations

import argparse
import base64
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rawpy
from PIL import Image


OUTPUT_DIR = Path("/Volumes/mac_ext/temp/raw_converted/")
SUPPORTED_EXTENSIONS = {".cr3", ".arw"}


def iter_raw_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_output_path(source_path: Path, output_dir: Path) -> Path:
    parent_name = source_path.parent.name or "root"
    output_name = f"{parent_name}__{source_path.stem}.jpg"
    return output_dir / output_name


def validate_raw_file(source_path: Path) -> Path:
    resolved_path = source_path.expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"RAW file does not exist: {resolved_path}")
    if resolved_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported RAW file type: {resolved_path}")
    return resolved_path


def convert_raw_file(source_path: Path, destination_path: Path) -> None:
    with rawpy.imread(str(source_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
        )

    image = Image.fromarray(rgb)
    image.save(destination_path, format="JPEG", quality=95, subsampling=0)


def convert_raw_to_jpg(
    source_path: Path, output_dir: Path = OUTPUT_DIR
) -> tuple[Path, bool]:
    raw_path = validate_raw_file(source_path)
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    destination_path = build_output_path(raw_path, resolved_output_dir)
    if destination_path.exists():
        return destination_path, False

    convert_raw_file(raw_path, destination_path)
    return destination_path, True


def jpg_file_to_base64(jpg_path: Path) -> str:
    with jpg_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def convert_raw_to_base64(source_path: Path, output_dir: Path = OUTPUT_DIR) -> str:
    jpg_path, _ = convert_raw_to_jpg(source_path, output_dir)
    return jpg_file_to_base64(jpg_path)


def convert_one_file(source_path: Path, output_dir: Path) -> tuple[str, Path, str | None]:
    try:
        destination_path, created = convert_raw_to_jpg(source_path, output_dir)
        if created:
            return "converted", destination_path, None
        return "skipped", destination_path, None
    except Exception as exc:
        return "failed", source_path, str(exc)


def render_progress(
    done_count: int,
    total_count: int,
    converted_count: int,
    skipped_count: int,
    failed_count: int,
) -> str:
    if total_count == 0:
        percent = 100.0
    else:
        percent = done_count / total_count * 100
    return (
        f"\rProgress {done_count}/{total_count} ({percent:5.1f}%) "
        f"converted={converted_count} skipped={skipped_count} failed={failed_count}"
    )


def convert_directory(
    input_dir: Path, output_dir: Path, workers: int, verbose: bool
) -> tuple[int, int, int]:
    raw_files = iter_raw_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    skipped_count = 0
    failed_count = 0
    done_count = 0
    total_count = len(raw_files)

    if not verbose:
        print(render_progress(0, total_count, 0, 0, 0), end="", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(convert_one_file, source_path, output_dir): source_path
            for source_path in raw_files
        }

        for future in as_completed(future_to_path):
            status, path, error_message = future.result()
            done_count += 1
            if status == "converted":
                converted_count += 1
                if verbose:
                    print(f"OK   {path}")
            elif status == "skipped":
                skipped_count += 1
                if verbose:
                    print(f"SKIP {path}")
            else:
                failed_count += 1
                if not verbose:
                    sys.stdout.write("\n")
                print(f"FAIL {path}: {error_message}")

            if not verbose:
                sys.stdout.write(
                    render_progress(
                        done_count,
                        total_count,
                        converted_count,
                        skipped_count,
                        failed_count,
                    )
                )
                sys.stdout.flush()

    if not verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return converted_count, skipped_count, failed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CR3 and ARW RAW photos to JPEG files."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing RAW files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for converted JPEG files. Defaults to {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads. Defaults to 8.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per file instead of a single-line progress display.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")
    if args.workers < 1:
        raise SystemExit("workers must be at least 1")

    converted_count, skipped_count, failed_count = convert_directory(
        input_dir, output_dir, args.workers, args.verbose
    )
    total_count = converted_count + skipped_count + failed_count

    print(
        "SUMMARY "
        f"total={total_count} converted={converted_count} skipped={skipped_count} failed={failed_count}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())