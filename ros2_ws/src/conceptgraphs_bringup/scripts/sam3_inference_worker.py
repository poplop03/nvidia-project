#!/usr/bin/env python3
"""SAM3 subprocess worker. Binary stdin/stdout protocol; no ROS dependency."""

import argparse
import base64
import json
import os
import struct
import sys

import cv2
import numpy as np


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main():
    # Reserve the original stdout exclusively for the binary protocol. Libraries
    # may print banners/progress; redirect those messages to stderr first.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from ultralytics.models.sam import SAM3SemanticPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts", required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mode", choices=("online_3d", "offline_2d"),
                        default="online_3d")
    args = parser.parse_args()
    prompts = args.prompts

    import torch

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    predictor = SAM3SemanticPredictor(overrides={
        "conf": args.conf, "task": "segment", "mode": "predict",
        "imgsz": args.imgsz, "model": args.model, "device": device,
        "half": device != "cpu", "save": False, "verbose": False,
    })
    source = sys.stdin.buffer
    sink = os.fdopen(protocol_fd, "wb", buffering=0)

    while True:
        raw_size = read_exact(source, 4)
        if raw_size is None:
            break
        header_size = struct.unpack("!I", raw_size)[0]
        header = json.loads(read_exact(source, header_size))
        height, width = header["height"], header["width"]
        rgb = np.frombuffer(read_exact(source, height * width * 3), np.uint8).reshape(height, width, 3)
        if args.mode == "online_3d":
            depth = np.frombuffer(read_exact(source, height * width * 2), np.uint16).reshape(height, width)
            k = np.asarray(header["camera_k"], dtype=np.float64).reshape(3, 3)
            world_from_camera = np.asarray(header["world_from_camera"], dtype=np.float64).reshape(4, 4)

        detections = []
        raw_masks = 0
        predictor.set_image(rgb)
        results = predictor(text=prompts)
        debug_image = rgb.copy()
        if results and args.mode == "online_3d":
            debug_image = results[0].plot(boxes=False, labels=True, conf=True)
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            raw_masks = len(masks)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            names = results[0].names
            for mask, confidence, class_id, box in zip(masks, confidences, class_ids, boxes):
                label = names[class_id] if isinstance(names, dict) else names[class_id]
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                if args.mode == "offline_2d":
                    binary = mask.astype(np.uint8) * 255
                    ok, encoded_mask = cv2.imencode(".png", binary)
                    if not ok:
                        continue
                    detections.append({
                        "confidence": float(confidence), "class_id": int(class_id),
                        "label": str(label),
                        "bbox_xyxy": [float(value) for value in box],
                        "mask_png": base64.b64encode(encoded_mask).decode(),
                        "mask_area_pixels": int(mask.sum()),
                    })
                    continue
                # Erosion removes depth contamination along mask boundaries.
                mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
                rows, cols = np.nonzero(mask & (depth > 0))
                z = depth[rows, cols].astype(np.float64) * header["depth_scale_m"]
                valid = (z > header.get("min_depth", 0.15)) & (z < header["max_depth"])
                rows, cols, z = rows[valid], cols[valid], z[valid]
                if len(z) < header["min_depth_samples"]:
                    continue
                median_z = np.median(z)
                near = np.abs(z - median_z) < max(0.20, median_z * 0.10)
                rows, cols, z = rows[near], cols[near], z[near]
                if len(z) < header["min_depth_samples"]:
                    continue
                camera_points = np.column_stack((
                    (cols - k[0, 2]) * z / k[0, 0],
                    (rows - k[1, 2]) * z / k[1, 1],
                    z,
                    np.ones_like(z),
                ))
                camera_center = np.median(camera_points[:, :3], axis=0)
                distance = float(np.linalg.norm(camera_center))
                if distance > header["max_distance"]:
                    continue
                world = (world_from_camera @ camera_points.T).T[:, :3]
                center = np.median(world, axis=0)
                detections.append({"x": float(center[0]), "y": float(center[1]),
                                   "z": float(center[2]), "confidence": float(confidence),
                                   "distance": distance, "samples": int(len(z)),
                                   "class_id": int(class_id), "label": str(label)})

        debug_jpeg = ""
        if args.mode == "online_3d":
            ok, encoded = cv2.imencode(".jpg", debug_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            debug_jpeg = base64.b64encode(encoded).decode() if ok else ""
        identifier = ({"stamp_ns": header["stamp_ns"]} if args.mode == "online_3d"
                      else {"view_id": header["view_id"]})
        payload = json.dumps({**identifier, "raw_masks": raw_masks,
                              "detections": detections, "debug_jpeg": debug_jpeg}).encode()
        sink.write(struct.pack("!I", len(payload)))
        sink.write(payload)
        sink.flush()


if __name__ == "__main__":
    main()
