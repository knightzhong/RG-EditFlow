#!/usr/bin/env python
import glob
import json
import os
import sys

paths = sys.argv[1:] or glob.glob("runs/**/*.json", recursive=True)
rows = []
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows.append({
        "path": path,
        "mean": data.get("final_mean"),
        "p100": data.get("p100"),
        "p80": data.get("p80"),
        "p50": data.get("p50"),
        "improvement": data.get("improvement"),
        "norm_mean": data.get("normalized_final_mean"),
        "norm_p100": data.get("normalized_p100"),
        "norm_p80": data.get("normalized_p80"),
        "norm_p50": data.get("normalized_p50"),
        "norm_improvement": data.get("normalized_improvement"),
        "quality": data.get("use_quality_gating"),
        "proposals": data.get("num_proposals"),
        "noise": data.get("proposal_noise_scale"),
        "rerank": data.get("rerank_mode"),
    })

rows.sort(key=lambda row: (row["mean"] is None, -(row["mean"] or -1)))
print("path,mean,p100,p80,p50,improvement,norm_mean,norm_p100,norm_p80,norm_p50,norm_improvement,quality,proposals,noise,rerank")
for row in rows:
    print(",".join(str(row[key]) for key in ["path", "mean", "p100", "p80", "p50", "improvement", "norm_mean", "norm_p100", "norm_p80", "norm_p50", "norm_improvement", "quality", "proposals", "noise", "rerank"]))
