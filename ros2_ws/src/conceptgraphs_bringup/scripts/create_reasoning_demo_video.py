#!/usr/bin/env python3
"""Create a 4x RGB-SAM/depth demo with five-second reasoning pauses."""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import cv2
import yaml


COLORS = {
    "blue_plastic_container": (255, 150, 30),
    "red_plastic_container": (50, 70, 255),
    "yellow_plastic_container": (40, 220, 255),
    "ladder": (230, 230, 230),
    "wheelbarrow": (180, 70, 255),
}


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fit_text(text, width=52):
    return textwrap.wrap(str(text), width=width, break_long_words=False) or [""]


def title(label):
    return label.replace("_", " ").upper()


def draw_text(frame, text, origin, scale=0.66, color=(245, 245, 245),
              thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color,
                thickness, cv2.LINE_AA)


def reasoning_text(caption, assessment):
    obj = caption["parsed"]["physical_objects"][0]
    physical = obj["physical_properties"]
    traversal = obj["object_traversability"]
    reasoning = (assessment or {}).get("reasoning_summary") or physical.get("evidence")
    return obj, physical, traversal, reasoning


def draw_popup(frame, event, elapsed, duration):
    caption = event["caption"]
    obj, physical, traversal, reasoning = reasoning_text(
        caption, event.get("assessment")
    )
    color = COLORS.get(event["canonical_label"], (70, 220, 255))
    x1, y1, x2, y2 = 525, 35, 1250, 455
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 15, 22), -1)
    cv2.addWeighted(overlay, 0.91, frame, 0.09, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    cv2.rectangle(frame, (x1, y1), (x2, y1 + 58), color, -1)
    draw_text(frame, "OBJECT REASONING", (x1 + 20, y1 + 39),
              0.75, (15, 18, 24), 2)

    y = y1 + 94
    draw_text(frame, title(event["canonical_label"]), (x1 + 22, y), 0.88, color, 2)
    y += 32
    draw_text(frame, f"{event['object_id']}  |  {event['aisle_id']}  |  {event['view_id']}",
              (x1 + 22, y), 0.53, (200, 210, 220))
    y += 34
    draw_text(frame, f"Appearance: {obj.get('color')}  |  {obj.get('material')}  |  {obj.get('state')}",
              (x1 + 22, y), 0.57)
    y += 30
    draw_text(frame, "Physical: " + ", ".join(filter(None, [
        physical.get("rigidity"),
        f"deformability {physical.get('deformability')}",
        f"mass {physical.get('estimated_mass_class')}",
    ])), (x1 + 22, y), 0.55)
    y += 30
    path_assessment = (event.get("assessment") or {}).get("passability", "unknown")
    draw_text(frame, f"Object: {title(traversal.get('semantic_class', 'unknown'))}  |  "
              f"Path: {title(path_assessment)}", (x1 + 22, y), 0.55, color, 2)
    y += 29
    draw_text(frame, f"Action: {title(traversal.get('contact_recommendation', 'unknown'))}  |  "
              f"navigation cost {str(traversal.get('navigation_cost', 'unknown')).upper()}",
              (x1 + 22, y), 0.54)
    y += 31
    draw_text(frame, "Reasoning:", (x1 + 22, y), 0.56, (180, 210, 255), 2)
    for line in fit_text(reasoning, 58)[:4]:
        y += 24
        draw_text(frame, line, (x1 + 22, y), 0.49, (225, 230, 235))

    progress = max(0.0, min(1.0, elapsed / duration))
    cv2.rectangle(frame, (x1, y2 - 8), (x2, y2), (55, 60, 68), -1)
    cv2.rectangle(frame, (x1, y2 - 8),
                  (x1 + int((x2 - x1) * progress), y2), color, -1)
    remaining = max(0.0, duration - elapsed)
    draw_text(frame, f"RESUMING IN {remaining:0.1f}s", (x2 - 188, y2 - 18),
              0.47, (210, 215, 220))


def add_normal_hud(frame, real_time, speed):
    # Replace the titles burned into the two source videos.
    cv2.rectangle(frame, (0, 0), (1280, 39), (12, 16, 14), -1)
    draw_text(frame, "SEGMENTATION", (24, 30), 0.82, (245, 245, 245), 2)
    cv2.rectangle(frame, (0, 480), (1280, 519), (12, 16, 14), -1)
    draw_text(frame, "DEPTH PERCEPTION", (24, 510), 0.82, (245, 245, 245), 2)
    cv2.rectangle(frame, (1045, 8), (1268, 43), (12, 16, 22), -1)
    draw_text(frame, f"{speed:g}x REAL-TIME  {real_time:06.1f}s", (1060, 33),
              0.52, (220, 230, 240), 1)
    cv2.line(frame, (0, 480), (1280, 480), (235, 235, 235), 3)


def load_events(dataset, bag_start_ns):
    views = {row["view_id"]: row for row in json_lines(dataset / "keyframes/views.jsonl")}
    objects = json_lines(dataset / "objects/objects.jsonl")
    captions = {row["object_id"]: row for row in json.loads(
        (dataset / "semantics/object_captions.json").read_text()
    )}
    assessments = {row["view_id"]: row for row in json.loads(
        (dataset / "semantics/traversability_assessments.json").read_text()
    )}
    events = []
    for item in objects:
        view_id = item["representative_view_id"]
        events.append({
            "object_id": item["object_id"],
            "canonical_label": item["canonical_label"],
            "aisle_id": item["aisle_id"],
            "view_id": view_id,
            "real_time_s": (views[view_id]["timestamp_ns"] - bag_start_ns) / 1e9,
            "caption": captions[item["object_id"]],
            "assessment": assessments.get(view_id),
        })
    events.sort(key=lambda row: row["real_time_s"])
    for index, event in enumerate(events, 1):
        event["ordinal"] = index
    return events


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read source frame {frame_index}")
    return cv2.resize(frame, (1280, 480), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sam-video", type=Path, required=True)
    parser.add_argument("--depth-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--pause-sec", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    pipeline = yaml.safe_load((dataset / "pipeline_config.yaml").read_text())
    bag = Path(pipeline["bag_path"])
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())[
        "rosbag2_bagfile_information"
    ]
    bag_start_ns = int(metadata["starting_time"]["nanoseconds_since_epoch"])
    bag_duration_s = int(metadata["duration"]["nanoseconds"]) / 1e9
    events = load_events(dataset, bag_start_ns)
    if len(events) != 6:
        raise RuntimeError(f"Expected 6 object events, found {len(events)}")

    sam = cv2.VideoCapture(str(args.sam_video))
    depth = cv2.VideoCapture(str(args.depth_video))
    if not sam.isOpened() or not depth.isOpened():
        raise RuntimeError("Could not open SAM or depth source video")
    source_fps = sam.get(cv2.CAP_PROP_FPS)
    source_frames = min(int(sam.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(depth.get(cv2.CAP_PROP_FRAME_COUNT)))
    source_duration_s = source_frames / source_fps
    recording_scale = source_duration_s / bag_duration_s

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x960",
        "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    base_frames = int(round((bag_duration_s / args.speed) * args.fps))
    pause_frames = int(round(args.pause_sec * args.fps))
    event_index = 0
    try:
        for output_index in range(base_frames):
            base_time_s = output_index / args.fps
            real_time_s = base_time_s * args.speed
            source_time_s = real_time_s * recording_scale
            source_index = min(source_frames - 1, int(round(source_time_s * source_fps)))
            top = read_frame(sam, source_index)
            bottom = read_frame(depth, source_index)
            combined = cv2.vconcat([top, bottom])
            add_normal_hud(combined, real_time_s, args.speed)

            while (event_index < len(events) and
                   real_time_s >= events[event_index]["real_time_s"]):
                event = events[event_index]
                for pause_index in range(pause_frames):
                    popup = combined.copy()
                    draw_popup(popup, event, pause_index / args.fps, args.pause_sec)
                    encoder.stdin.write(popup.tobytes())
                event_index += 1
            encoder.stdin.write(combined.tobytes())
    finally:
        sam.release()
        depth.release()
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    report = {
        "output": str(args.output.resolve()),
        "resolution": [1280, 960],
        "fps": args.fps,
        "speed_vs_real_time": args.speed,
        "pause_sec_per_object": args.pause_sec,
        "object_events": [
            {key: event[key] for key in (
                "ordinal", "object_id", "canonical_label", "aisle_id",
                "view_id", "real_time_s"
            )} for event in events
        ],
        "expected_duration_s": bag_duration_s / args.speed + len(events) * args.pause_sec,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
