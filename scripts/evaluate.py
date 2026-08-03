#!/usr/bin/env python3
"""Strict top-1 RefSegRS-style evaluation for RefSegRS and RRSIS-D."""

import argparse
import json
import os

import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def decode_rle(rle):
    if not rle:
        return None
    if not rle.get("counts"):
        return np.zeros(rle.get("size", [0, 0]), dtype=bool)
    encoded = dict(rle)
    if isinstance(encoded["counts"], str):
        encoded["counts"] = encoded["counts"].encode("utf-8")
    return mask_utils.decode(encoded).astype(bool)


def index_unique(records, key, description):
    indexed = {}
    for record in records:
        value = record[key]
        if value in indexed:
            raise ValueError(f"duplicate {description}: {value}")
        indexed[value] = record
    return indexed


def evaluate(ground_truth, predictions):
    gt_map = index_unique(
        ground_truth["annotations"], "id", "ground-truth annotation id"
    )
    pred_map = index_unique(
        predictions, "annotation_id", "prediction annotation id"
    )

    gt_ids = set(gt_map)
    pred_ids = set(pred_map)
    missing = sorted(gt_ids - pred_ids)
    extra = sorted(pred_ids - gt_ids)
    if missing or extra:
        raise ValueError(
            "prediction/GT annotation ids are not identical: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    ious = []
    intersection_sum = 0.0
    union_sum = 0.0

    for annotation_id in tqdm(sorted(gt_ids), desc="Evaluating"):
        gt_mask = decode_rle(gt_map[annotation_id]["segmentation"])
        if gt_mask is None:
            raise ValueError(f"annotation {annotation_id} has no GT mask")

        candidates = pred_map[annotation_id].get("predictions", [])
        if candidates:
            top = max(candidates, key=lambda item: item.get("score", 0.0))
            pred_mask = decode_rle(top["rle"])
        else:
            pred_mask = np.zeros_like(gt_mask, dtype=bool)

        if pred_mask is None:
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"mask shape mismatch for annotation {annotation_id}: "
                f"prediction={pred_mask.shape}, GT={gt_mask.shape}"
            )

        intersection = float(np.logical_and(pred_mask, gt_mask).sum())
        union = float(np.logical_or(pred_mask, gt_mask).sum())
        iou = intersection / (union + 1e-5)

        intersection_sum += intersection
        union_sum += union
        ious.append(iou)

    if not ious:
        raise ValueError("no samples were evaluated")

    metrics = {
        f"P@{threshold:.1f}": 100.0
        * float(np.mean([iou > threshold for iou in ious]))
        for threshold in THRESHOLDS
    }
    metrics["cIoU"] = 100.0 * intersection_sum / (union_sum + 1e-10)
    metrics["gIoU"] = 100.0 * float(np.mean(ious))
    metrics["samples"] = len(ious)
    return metrics


def print_report(metrics):
    width = 60
    print("\n" + "=" * width)
    print("  RRSIS TOP-1 EVALUATION REPORT")
    print("=" * width)
    print(f"Total Samples: {metrics['samples']}")
    print("-" * width)
    print(f"{'Metric':<30} | {'Score (%)':<10}")
    print("-" * width)
    for threshold in THRESHOLDS:
        name = f"P@{threshold:.1f}"
        print(f"{name:<30} | {metrics[name]:.2f}")
    print(f"{'cIoU':<30} | {metrics['cIoU']:.2f}")
    print(f"{'gIoU':<30} | {metrics['gIoU']:.2f}")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Strict annotation-aligned top-1 RRSIS evaluation"
    )
    parser.add_argument("--gt", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    with open(args.gt, "r", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    with open(args.predictions, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    metrics = evaluate(ground_truth, predictions)
    print_report(metrics)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()

