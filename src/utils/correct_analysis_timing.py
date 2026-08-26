#!/usr/bin/env python3
"""Re-encode recorded videos on their real wall-clock timeline.

MP4 files are encoded at a fixed requested FPS even when rendering cannot
maintain that rate. This utility resamples frames onto a fixed wall-clock grid:
slow-frame gaps hold the latest frame, while catch-up bursts discard redundant
intermediate frames. New recordings use ``video_timestamps.csv`` and the
retained ``analysis_capture.mp4``; trajectory timestamps and ``analysis.mp4``
remain fallbacks for older recordings.

Usage:
    python src/utils/correct_analysis_timing.py experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path


def read_timestamps(path: Path) -> list[float]:
    with path.open(newline="") as stream:
        values = [float(row["t"]) for row in csv.DictReader(stream)]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: timestamps must be non-empty and finite")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"{path}: timestamps must be nondecreasing")
    return values


def _source_indices(timestamps: list[float], fps: float) -> list[int]:
    """Return the latest captured frame available at each output clock tick."""
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("output FPS must be finite and positive")
    if not timestamps:
        raise ValueError("timestamps must be non-empty")
    relative = [value - timestamps[0] for value in timestamps]
    last_tick = int(math.ceil(relative[-1] * fps - 1e-9))
    source_idx = 0
    indices = []
    for tick in range(last_tick + 1):
        output_time = tick / fps
        while (
            source_idx + 1 < len(relative)
            and relative[source_idx + 1] <= output_time + 1e-9
        ):
            source_idx += 1
        indices.append(source_idx)
    return indices


def _configured_fps(run_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return 10.0
    return float(json.loads(metadata_path.read_text()).get("fps", 10.0))


def reencode_video(
    source: Path,
    output: Path,
    timestamps: list[float],
    fps: float,
    *,
    overwrite: bool = False,
    crf: int = 10,
    preset: str = "veryfast",
) -> tuple[int, int, float]:
    import cv2

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists (use --overwrite)")

    indices = _source_indices(timestamps, fps)
    capture = cv2.VideoCapture(str(source))
    ok, current = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"cannot decode first frame of {source}")

    height, width = current.shape[:2]
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    decoded = 1
    current_idx = 0
    try:
        for wanted_idx in indices:
            while current_idx < wanted_idx:
                ok, current = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"{source}: decoded {decoded} frames for "
                        f"{len(timestamps)} timestamps"
                    )
                decoded += 1
                current_idx += 1
            process.stdin.write(current.tobytes())

        while capture.grab():
            decoded += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed for {source}: {stderr.strip()}")
        if decoded != len(timestamps):
            raise RuntimeError(
                f"{source}: decoded {decoded} frames but found "
                f"{len(timestamps)} timestamps"
            )
        temporary.replace(output)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    duration = len(indices) / fps
    return decoded, len(indices), duration


def _reencode(run_dir: Path, fps: float, overwrite: bool) -> tuple[int, int, float]:
    timestamp_path = run_dir / "video_timestamps.csv"
    if not timestamp_path.exists():
        timestamp_path = run_dir / "trajectory.csv"
    source = run_dir / "analysis_capture.mp4"
    if not source.exists():
        source = run_dir / "analysis.mp4"
    return reencode_video(
        source,
        run_dir / "analysis_corrected.mp4",
        read_timestamps(timestamp_path),
        fps,
        overwrite=overwrite,
    )


def _self_test() -> None:
    assert _source_indices([0.02, 0.12, 0.42], 10.0) == [0, 1, 1, 1, 2]
    assert _source_indices([0.0, 0.04, 0.08, 0.20], 10.0) == [0, 2, 3]
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("experiments"))
    parser.add_argument("--fps", type=float, help="override metadata output FPS")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")

    runs = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir()
        and (path / "analysis.mp4").exists()
        and (path / "trajectory.csv").exists()
    )
    if not runs:
        parser.error(f"no recorded runs found under {args.root}")

    failures = []
    for run_dir in runs:
        try:
            fps = _configured_fps(run_dir, args.fps)
            source_frames, output_frames, duration = _reencode(
                run_dir, fps, args.overwrite
            )
            print(
                f"{run_dir.name}: {source_frames} source frames -> "
                f"{output_frames} frames, {duration:.2f}s at {fps:g} FPS"
            )
        except Exception as error:
            failures.append(run_dir.name)
            print(f"{run_dir.name}: ERROR: {error}")
    if failures:
        print(f"failed runs: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
