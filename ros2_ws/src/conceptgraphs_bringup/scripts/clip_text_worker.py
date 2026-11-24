#!/usr/bin/env python3
"""Encode one text query with the same OpenAI CLIP model used by M6."""
import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", default="ViT-B-16")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

name = {"ViT-B-16": "ViT-B/16", "ViT-B-32": "ViT-B/32"}.get(args.model, args.model)
model, _ = clip.load(name, device=args.device, jit=False)
model.eval()
with torch.no_grad():
    tokens = clip.tokenize([args.text]).to(args.device)
    vector = model.encode_text(tokens).float()
    vector /= vector.norm(dim=-1, keepdim=True)
array = vector.cpu().numpy().astype("float32")[0]
args.output.parent.mkdir(parents=True, exist_ok=True)
np.save(args.output, array)
print(json.dumps({"model": args.model, "dimension": int(array.shape[0])}))
