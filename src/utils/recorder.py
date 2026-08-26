"""Demo/experiment recorder for real-world runs.

One run → experiments/<timestamp>_<target>/:
    metadata.json   target, mode, git sha, start time, fps
    rgb.mp4         upright camera video (the watchable demo)
    obstacle_map.mp4 synchronized top-down obstacle/path map
    analysis.mp4    timestamped RGB + obstacle-map side-by-side video
    *_capture.mp4   original synchronized frames retained for reprocessing
    video_timestamps.csv exact capture time for every synchronized video frame
    trajectory.csv  robot pose over time (t, x, y, yaw)
    goals.jsonl     each distinct decision goal (t, x, y, source)
    events.jsonl    timestamped milestones (goal, target_found, done)
    final/topdown.png, scene_graph.json
    result.json     outcome + metrics (found, time_to_find, path_length, n_goals)

Samples shared_state on its own thread (frames stream straight to disk, nothing
buffered in memory), so it neither slows the nav loop nor grows unbounded across
back-to-back runs. All I/O is best-effort — a recorder failure never aborts the
demo. The run dir is uniquified, so two runs in the same second don't collide.
"""
import csv
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class _HighQualityVideoWriter:
    """Raw BGR frames to high-quality H.264, with an OpenCV fallback."""

    def __init__(self, path, fps, size, *, crf=14, preset="veryfast"):
        self._fallback = None
        self._process = None
        width, height = size
        try:
            self._process = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
                    "-an", "-c:v", "libx264", "-preset", str(preset),
                    "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, ValueError):
            self._fallback = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._fallback is not None:
            self._fallback.write(frame)
            return
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass

    def release(self):
        if self._fallback is not None:
            self._fallback.release()
            return
        if self._process is None:
            return
        try:
            self._process.stdin.close()
            self._process.wait(timeout=10.0)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self._process.kill()
            self._process.wait()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _unique_dir(root, name):
    d = Path(root) / name
    i = 2
    while d.exists():
        d = Path(root) / f"{name}_{i}"
        i += 1
    return d


class DemoRecorder:
    def __init__(self, shared_state, target, camera_name,
                 root="experiments", fps=10, rgb_rotation_k=0,
                 topdown_renderer=None, planner_debug_getter=None):
        self.ss = shared_state
        self.camera_name = camera_name
        self.fps = fps
        self.rgb_rotation_k = rgb_rotation_k
        self._topdown_renderer = topdown_renderer
        self._planner_debug_getter = planner_debug_getter
        ts = time.strftime("%Y%m%d_%H%M%S")
        slug = (target or "explore").strip().replace(" ", "-") or "explore"
        self.dir = _unique_dir(root, f"{ts}_{slug}")
        (self.dir / "final").mkdir(parents=True, exist_ok=True)
        self._monotonic_t0 = time.monotonic()

        self._writer = None
        self._map_writer = None
        self._analysis_writer = None
        self._vlm_frame = None
        self._vlm_frame_path = None
        self._det_viz = None
        self._det_frame = None
        self._det_frame_key = None
        self._analysis_panel_widths = None
        self._video_frames = 0
        self._unique_rgb_frames = 0
        self._last_rgb_step_id = None
        self._traj_f = open(self.dir / "trajectory.csv", "w", newline="")
        self._traj = csv.writer(self._traj_f)
        self._traj.writerow(["t", "x", "y", "yaw"])
        self._video_times_f = open(
            self.dir / "video_timestamps.csv", "w", newline="", buffering=1
        )
        self._video_times = csv.writer(self._video_times_f)
        self._video_times.writerow([
            "frame", "t", "step_id", "observation_timestamp"
        ])
        self._events_f = open(self.dir / "events.jsonl", "w", buffering=1)
        self._goals_f = open(self.dir / "goals.jsonl", "w", buffering=1)
        self._planner_f = open(self.dir / "planner_debug.jsonl", "w", buffering=1)

        self._last_goal = None
        self._n_goals = 0
        self._path_len = 0.0
        self._last_xy = None
        self._found_t = None
        self._stop = threading.Event()
        self._thread = None

        (self.dir / "metadata.json").write_text(json.dumps({
            "target": target, "mode": "full_auto", "git": _git_sha(),
            "start": ts, "fps": fps,
        }, indent=2))
        print(f"[recorder] → {self.dir}")

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def event(self, kind, **data):
        self._events_f.write(json.dumps(
            {"t": round(self._elapsed(), 2), "kind": kind, **data}) + "\n")

    def mark_found(self):
        if self._found_t is None:
            self._found_t = round(self._elapsed(), 2)
        self.event("target_found", t_find=self._found_t)

    def finalize(self, found):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        duration = self._elapsed()
        if self._writer is not None:
            self._writer.release()
        if self._map_writer is not None:
            self._map_writer.release()
        if self._analysis_writer is not None:
            self._analysis_writer.release()
        self._traj_f.close()
        self._video_times_f.close()
        timing_corrected = self._finalize_videos()
        self._dump("final/scene_graph.json", self._scene_graph())
        self._dump_topdown()
        result = {
            "found": bool(found),
            "duration_s": round(duration, 1),
            "time_to_find_s": self._found_t,
            "path_length_m": round(self._path_len, 2),
            "n_goals": self._n_goals,
            "video_frames": self._video_frames,
            "recorded_fps": round(self._video_frames / max(duration, 1e-6), 2),
            "unique_camera_frames": self._unique_rgb_frames,
            "unique_camera_fps": round(
                self._unique_rgb_frames / max(duration, 1e-6), 2
            ),
            "video_timing_corrected": timing_corrected,
            "raw_video_retained": all(
                (self.dir / f"{name}_capture.mp4").exists()
                for name in ("rgb", "obstacle_map", "analysis")
            ) if self._video_frames else None,
        }
        self._dump("result.json", result)
        self._events_f.close()
        self._goals_f.close()
        self._planner_f.close()
        print(f"[recorder] saved {self.dir} ({result})")
        return result

    # --- sampler ---------------------------------------------------------
    def _loop(self):
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as e:
                print(f"[recorder] sample error: {e}")
            next_tick += period
            now = time.monotonic()
            if next_tick < now:
                next_tick = now
            self._stop.wait(max(0.0, next_tick - now))

    def _sample(self):
        elapsed_s = self._elapsed()
        t = round(elapsed_s, 2)
        with self.ss.lock:
            obs = getattr(self.ss.sensor, "habitat_obs", None)
            rgb = np.asarray(obs.rgb) if (obs is not None and obs.rgb is not None) else None
            step_id = getattr(obs, "step_id", None) if obs is not None else None
            observation_timestamp = (
                getattr(obs, "timestamp", None) if obs is not None else None
            )
            odom = getattr(self.ss.sensor, "latest_odom", None)
            control_odom = getattr(self.ss.sensor, "control_odom", None)
            control_odom_meta = getattr(
                self.ss.sensor, "control_odom_meta", None
            )
            goal = getattr(self.ss.nav, "goal_xy", None)
            src = getattr(self.ss.nav, "goal_source", None)
            nav_status = getattr(self.ss.nav, "status", None)
            detection_result = getattr(self.ss.perception, "detection_result", None)
            target_labels = set(getattr(self.ss.task, "synonym_to_canonical", {}) or {})
            obstacle_map = getattr(self.ss.mapping, "obstacle_map", None)

        if odom is not None:
            self._traj.writerow([t, round(odom[0], 3), round(odom[1], 3), round(odom[2], 3)])
            if self._last_xy is not None:
                self._path_len += float(np.hypot(odom[0] - self._last_xy[0],
                                                 odom[1] - self._last_xy[1]))
            self._last_xy = (odom[0], odom[1])
        self._record_planner_debug(
            t, odom, control_odom, control_odom_meta,
            goal, src, nav_status, obstacle_map
        )
        map_frame = self._render_topdown()
        if rgb is not None and map_frame is not None:
            self._write_video_pair(
                rgb, map_frame, elapsed_s, step_id, observation_timestamp,
                detection_result, target_labels,
            )
            if step_id != self._last_rgb_step_id:
                self._unique_rgb_frames += 1
                self._last_rgb_step_id = step_id
        if goal is not None:
            key = (round(float(goal[0]), 2), round(float(goal[1]), 2), src)
            if key != self._last_goal:
                self._last_goal = key
                self._n_goals += 1
                self._goals_f.write(json.dumps(
                    {"t": t, "x": key[0], "y": key[1], "source": src}) + "\n")

    def _record_planner_debug(
            self, t, odom, control_odom, control_odom_meta,
            goal, source, status, obstacle_map):
        if self._planner_debug_getter is None or odom is None:
            return
        planner = self._planner_debug_getter()
        if planner is None:
            return
        path = getattr(planner, "path_xy", None)
        path = np.asarray(path, dtype=float) if path is not None else np.empty((0, 2))
        if len(path) > 120:
            path = path[np.linspace(0, len(path) - 1, 120, dtype=int)]
        tracker = getattr(planner, "tracker", None)
        command = getattr(tracker, "command", None)
        target_xy = getattr(tracker, "target_xy", None)
        row = {
            "t": t,
            "pose": [round(float(v), 4) for v in odom[:3]],
            "control_pose": (
                None if control_odom is None else
                [round(float(v), 4) for v in control_odom[:3]]
            ),
            "odom_timing": self._odom_timing(control_odom_meta, tracker),
            "goal": None if goal is None else [round(float(v), 4) for v in goal[:2]],
            "source": source,
            "status": status,
            "path": np.round(path[:, :2], 4).tolist(),
            "path_length_m": (
                round(float(np.linalg.norm(
                    np.diff(path[:, :2], axis=0), axis=1
                ).sum()), 4)
                if len(path) > 1 else 0.0
            ),
            "short_term_goal": self._xy(getattr(planner, "short_term_goal", None)),
            "lookahead": self._xy(target_xy),
            "heading_error_rad": (
                None if tracker is None else
                round(float(getattr(tracker, "heading_error", 0.0)), 4)
            ),
            "lateral_error_m": (
                None if tracker is None else
                round(float(getattr(tracker, "lateral_error", 0.0)), 4)
            ),
            "path_curvature_inv_m": (
                None if tracker is None else
                round(float(getattr(tracker, "path_curvature", 0.0)), 4)
            ),
            "desired_wz": (
                None if tracker is None else
                round(float(getattr(tracker, "desired_yaw_rate", 0.0)), 4)
            ),
            "measured_wz": (
                None if tracker is None else
                round(float(getattr(tracker, "measured_yaw_rate", 0.0)), 4)
            ),
            "open_lookahead": (
                None if tracker is None else
                bool(getattr(tracker, "_open_lookahead", False))
            ),
            "command": None if command is None else {
                "vx": round(float(command[0]), 4),
                "wz": round(float(command[1]), 4),
                "mode": command[2],
            },
        }
        if obstacle_map is not None:
            px = obstacle_map.xy_to_px(np.asarray([odom[:2]], dtype=float))[0]
            x, y = int(px[0]), int(px[1])
            row["robot_cell"] = {
                "x": x,
                "y": y,
                "explored": int(obstacle_map.explored[y, x]),
                "navigable": int(obstacle_map.navigable[y, x]),
                "occupied": int(obstacle_map.occupancy[y, x]),
            }
        self._planner_f.write(json.dumps(row, separators=(",", ":")) + "\n")

    @staticmethod
    def _odom_timing(meta, tracker):
        if meta is None:
            return None
        now_wall_ns = time.time_ns()
        source_ns = int(meta["source_stamp_ns"])
        arrival_ns = int(meta["arrival_wall_ns"])
        read_ns = getattr(tracker, "odom_read_wall_ns", None)
        tracker_age = getattr(tracker, "odom_callback_age_ms", None)
        return {
            "source_stamp_ns": source_ns,
            "callback_arrival_ns": arrival_ns,
            "tracker_read_ns": read_ns,
            "callback_sequence": int(meta["callback_sequence"]),
            "transport_age_ms": round((arrival_ns - source_ns) / 1e6, 3),
            "recorder_age_ms": round((now_wall_ns - arrival_ns) / 1e6, 3),
            "tracker_callback_age_ms": (
                None if tracker_age is None else round(float(tracker_age), 3)
            ),
            "tracker_sequence": (
                None if tracker is None else
                getattr(tracker, "odom_sequence", None)
            ),
        }

    @staticmethod
    def _xy(value):
        return None if value is None else [round(float(v), 4) for v in value[:2]]

    def _write_video_pair(
        self, rgb, map_frame, elapsed_s, step_id, observation_timestamp,
        detection_result, target_labels,
    ):
        if self.rgb_rotation_k:
            rgb = np.rot90(rgb, k=self.rgb_rotation_k)
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if self._writer is None:
            h, w = bgr.shape[:2]
            self._writer = _HighQualityVideoWriter(
                self.dir / "rgb_capture.mp4", self.fps, (w, h))
        map_bgr = np.ascontiguousarray(map_frame)
        if map_bgr.ndim == 2:
            map_bgr = cv2.cvtColor(map_bgr, cv2.COLOR_GRAY2BGR)
        elif map_bgr.shape[2] == 4:
            map_bgr = cv2.cvtColor(map_bgr, cv2.COLOR_BGRA2BGR)
        if self._map_writer is None:
            mh, mw = map_bgr.shape[:2]
            self._map_size = (mw, mh)
            self._map_writer = _HighQualityVideoWriter(
                self.dir / "obstacle_map_capture.mp4", self.fps, self._map_size)
        elif (map_bgr.shape[1], map_bgr.shape[0]) != self._map_size:
            map_bgr = cv2.resize(map_bgr, self._map_size, interpolation=cv2.INTER_AREA)
        self._writer.write(bgr)
        self._map_writer.write(map_bgr)
        detection_bgr = self._render_detections(detection_result, target_labels)
        analysis = self._analysis_frame(
            detection_bgr if detection_bgr is not None else bgr,
            map_bgr,
            elapsed_s,
        )
        if self._analysis_writer is None:
            ah, aw = analysis.shape[:2]
            self._analysis_writer = _HighQualityVideoWriter(
                self.dir / "analysis_capture.mp4", self.fps, (aw, ah),
                crf=10, preset="veryfast")
        self._analysis_writer.write(analysis)
        self._video_times.writerow([
            self._video_frames,
            f"{elapsed_s:.6f}",
            "" if step_id is None else step_id,
            "" if observation_timestamp is None else f"{observation_timestamp:.9f}",
        ])
        self._video_frames += 1

    def _finalize_videos(self):
        """Resample capture-rate frames onto the requested wall-clock timeline."""
        captures = (
            ("rgb", 14),
            ("obstacle_map", 14),
            ("analysis", 10),
        )
        if self._video_frames == 0:
            return None
        corrected = []
        try:
            try:
                from correct_analysis_timing import read_timestamps, reencode_video
            except ImportError:
                from .correct_analysis_timing import read_timestamps, reencode_video

            timestamps = read_timestamps(self.dir / "video_timestamps.csv")
            print("[recorder] correcting video timing...")
            for name, crf in captures:
                source = self.dir / f"{name}_capture.mp4"
                output = self.dir / f".{name}_corrected.mp4"
                final = self.dir / f"{name}.mp4"
                corrected.append((output, final))
                reencode_video(
                    source, output, timestamps, self.fps,
                    overwrite=True, crf=crf,
                )
            for output, final in corrected:
                output.replace(final)
            return True
        except Exception as error:
            print(f"[recorder] timing correction failed: {error}")
            for output, _ in corrected:
                output.unlink(missing_ok=True)
            for name, _ in captures:
                capture = self.dir / f"{name}_capture.mp4"
                output = self.dir / f"{name}.mp4"
                if capture.exists() and not output.exists():
                    shutil.copy2(capture, output)
            return False

    def _elapsed(self):
        return time.monotonic() - self._monotonic_t0

    def _analysis_frame(self, rgb_bgr, map_bgr, elapsed_s):
        panel_h = map_bgr.shape[0]
        header_h = 48
        body_h = panel_h - header_h
        vlm = self._latest_vlm_frame()
        if self._analysis_panel_widths is None:
            self._analysis_panel_widths = tuple(
                self._natural_panel_width(image, body_h)
                for image in (rgb_bgr, map_bgr, vlm if vlm is not None else map_bgr)
            )
        widths = self._analysis_panel_widths
        panels = [
            self._make_panel(rgb_bgr, widths[0], panel_h, header_h),
            self._make_panel(map_bgr, widths[1], panel_h, header_h),
            self._make_panel(vlm, widths[2], panel_h, header_h),
        ]
        gap = np.full((panel_h, 6, 3), 245, dtype=np.uint8)
        frame = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
        middle_x = widths[0] + gap.shape[1]
        right_x = middle_x + widths[1] + gap.shape[1]
        cv2.putText(frame, "OWLv2 DETECTIONS", (12, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "OBSTACLE MAP", (middle_x + 12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "VLM DECISION", (right_x + 12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        time_text = f"t={elapsed_s:.2f}s"
        (time_w, time_h), baseline = cv2.getTextSize(
            time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        )
        time_x = 12
        time_y = panel_h - 14
        cv2.rectangle(
            frame,
            (time_x - 6, time_y - time_h - 6),
            (time_x + time_w + 6, time_y + baseline + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame, time_text, (time_x, time_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
        )
        return frame

    @staticmethod
    def _natural_panel_width(image, height):
        width = round(height * image.shape[1] / image.shape[0])
        return max(2, width + width % 2)

    @classmethod
    def _make_panel(cls, image, width, height, header_h):
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        if image is not None:
            panel[header_h:] = cls._fit_panel(image, (width, height - header_h))
        return panel

    @staticmethod
    def _fit_panel(image, size):
        """Letterbox an image without changing its aspect ratio."""
        width, height = size
        scale = min(width / image.shape[1], height / image.shape[0])
        fitted = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)),
             max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        x = (width - fitted.shape[1]) // 2
        y = (height - fitted.shape[0]) // 2
        panel[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
        return panel

    def _render_detections(self, detection_result, target_labels):
        if detection_result is None:
            return self._det_frame
        key = (
            getattr(detection_result, "timestamp", None),
            getattr(detection_result, "episode_index", None),
            getattr(detection_result, "total_detections", None),
            tuple(sorted(str(label) for label in target_labels)),
        )
        if key == self._det_frame_key:
            return self._det_frame
        from rtnav.tools.visualization.detection_visualizer_thread import (
            WebDetectionVisualizerThread,
            _as_rgb3,
        )
        if self._det_viz is None:
            self._det_viz = WebDetectionVisualizerThread(
                self.ss, threading.Event(), display_height=0
            )
        cameras = getattr(detection_result, "camera_results", {}) or {}
        cam = cameras.get(self.camera_name)
        if cam is None and cameras:
            cam = cameras[sorted(cameras)[0]]
        frame = None
        if cam is not None:
            rgb = getattr(cam, "rgb_image_detector_input", None)
            detections = getattr(cam, "detections_detector_input", None)
            if rgb is None:
                rgb = getattr(cam, "rgb_image", None)
                detections = getattr(cam, "detections", None)
            if rgb is not None:
                frame = cv2.cvtColor(
                    _as_rgb3(rgb), cv2.COLOR_RGB2BGR
                )
                frame = self._det_viz._draw_detections_on_image(
                    frame, list(detections or []), target_labels
                )
        if frame is not None:
            self._det_frame = frame
            self._det_frame_key = key
        return self._det_frame

    def _latest_vlm_frame(self):
        root = self.dir / "vlm_decisions"
        paths = [
            *root.glob("frontier/*_selected.jpg"),
            *root.glob("verification/*.png"),
        ]
        if not paths:
            return self._vlm_frame
        path = max(paths, key=lambda item: item.stat().st_mtime_ns)
        if path != self._vlm_frame_path:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                self._vlm_frame = image
                self._vlm_frame_path = path
        return self._vlm_frame

    def _render_topdown(self):
        try:
            if self._topdown_renderer is not None:
                return self._topdown_renderer()
            from rtnav.tools.visualization.map_visualizer_thread import MapVisualizerBase

            return MapVisualizerBase(self.ss).render_decision_frame_clean()
        except Exception as e:
            print(f"[recorder] topdown sample failed: {e}")
            return None

    # --- final artifacts -------------------------------------------------
    def _dump(self, rel, obj):
        (self.dir / rel).write_text(json.dumps(obj, indent=2))

    def _dump_topdown(self):
        try:
            img = self._render_topdown()
            if img is not None and img.size > 0:
                cv2.imwrite(str(self.dir / "final/topdown.png"), img)
        except Exception as e:
            print(f"[recorder] topdown failed: {e}")

    def _scene_graph(self):
        with self.ss.lock:
            sg = self.ss.scenegraph.scene_graph
        nodes = list(getattr(sg, "nodes", []) or []) if sg is not None else []
        out = []
        for n in nodes:
            c = getattr(n, "centroid", None)
            if c is None:
                continue
            out.append({"label": getattr(n, "chosen_label", ""),
                        "centroid": [float(v) for v in np.asarray(c)[:3]]})
        return out
