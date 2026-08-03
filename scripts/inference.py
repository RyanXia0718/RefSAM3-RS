#!/usr/bin/env python3
"""Minimal full-model inference for RefSegRS and RRSIS-D simple queries."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SAM3_PKG = os.path.join(PROJECT_ROOT, "sam3")
if SAM3_PKG not in sys.path:
    sys.path.insert(0, SAM3_PKG)

from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


RESOLUTION = 1008
NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
TEXT_ADAPTER = {"adapter_dim": 64, "adapter_heads": 4, "adapter_scale": 1.0}
VISUAL_LORA = {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
TEXTSCALE = {
    "enabled": True,
    "rank": 64,
    "num_heads": 4,
    "mixer_kernels": [1, 3, 5],
}
BPE_PATH = os.path.join(SAM3_PKG, "assets", "bpe_simple_vocab_16e6.txt.gz")


@dataclass
class QueryItem:
    annotation_id: int
    image_id: int
    file_name: str
    prompt: Dict[str, Any]


class SimpleQueryDataset(Dataset):
    """One simple referring expression per annotation."""

    def __init__(self, json_path: str, image_root: str, transform):
        self.image_root = image_root
        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        images = {image["id"]: image for image in data["images"]}
        if len(images) != len(data["images"]):
            raise ValueError("ground-truth JSON contains duplicate image ids")

        self.items: List[QueryItem] = []
        seen_annotations = set()
        for annotation in data["annotations"]:
            annotation_id = annotation["id"]
            if annotation_id in seen_annotations:
                raise ValueError(f"duplicate annotation id: {annotation_id}")
            seen_annotations.add(annotation_id)

            image = images.get(annotation["image_id"])
            if image is None:
                raise ValueError(
                    f"annotation {annotation_id} references missing image "
                    f"{annotation['image_id']}"
                )

            text_input = image.get("text_inst_input", {})
            simple_queries = text_input.get("simple_query", [])
            if not simple_queries:
                raise ValueError(
                    f"annotation {annotation_id} has no simple_query"
                )

            prompt = {
                "simple_query": [simple_queries[0], simple_queries[0]],
            }
            self.items.append(
                QueryItem(
                    annotation_id=annotation_id,
                    image_id=annotation["image_id"],
                    file_name=image["file_name"],
                    prompt=prompt,
                )
            )

        print(f"Loaded {len(self.items)} simple-query samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image_path = os.path.join(self.image_root, item.file_name)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        datapoint = Datapoint(
            find_queries=[],
            images=[SAMImage(data=image, objects=[], size=[height, width])],
        )
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=item.prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=index,
                    original_image_id=item.image_id,
                    original_category_id=1,
                    original_size=[height, width],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        return self.transform(datapoint), item


def collate_batch(samples):
    datapoints = [sample[0] for sample in samples]
    items = [sample[1] for sample in samples]
    batch = collate(datapoints, dict_key="batch")["batch"]
    return batch, items


def build_transform():
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=RESOLUTION,
                max_size=RESOLUTION,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(
                mean=NORMALIZATION["mean"],
                std=NORMALIZATION["std"],
            ),
        ]
    )


def encode_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle, list):
        rle = rle[0]
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": list(rle["size"]), "counts": rle["counts"]}


def postprocess_batch(output, batch, items, postprocessor):
    """Keep exactly the highest-score mask for each referring expression."""
    processed = postprocessor.process_results(output, batch.find_metadatas)
    metadata = batch.find_metadatas[0]
    coco_ids = metadata.coco_image_id
    results = []

    for index, item in enumerate(items):
        if isinstance(coco_ids, torch.Tensor):
            key = (
                int(coco_ids[index])
                if coco_ids.ndim > 0
                else int(coco_ids.item())
            )
        else:
            key = int(coco_ids)

        predictions = []
        detection = processed.get(key)
        if detection is not None and len(detection["masks"]) > 0:
            masks = detection["masks"]
            scores = detection["scores"]
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.float().detach().cpu().numpy()

            if masks.ndim == 4:
                masks = masks.squeeze(1)
            best_index = int(np.argmax(scores))
            best_mask = masks[best_index]
            if best_mask.dtype != np.bool_:
                best_mask = best_mask > 0.5
            predictions.append(
                {
                    "rle": encode_mask(best_mask),
                    "score": float(scores[best_index]),
                }
            )

        results.append(
            {
                "annotation_id": item.annotation_id,
                "image_id": item.image_id,
                "file_name": item.file_name,
                "prompt": item.prompt,
                "prompt_type": "simple_query_0",
                "predictions": predictions,
            }
        )
    return results


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RefSAM3-RS inference")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"invalid --gpu {args.gpu}; visible CUDA device count is "
            f"{torch.cuda.device_count()}"
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    model = build_sam3_image_model(
        bpe_path=BPE_PATH if os.path.exists(BPE_PATH) else None,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        enable_segmentation=True,
        device="cuda",
        adapter_config=TEXT_ADAPTER,
        visual_lora_config=VISUAL_LORA,
        textscale_config=TEXTSCALE,
    ).to(device)
    model.eval()

    dataset = SimpleQueryDataset(
        json_path=args.annotations,
        image_root=args.image_root,
        transform=build_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.0,
        to_cpu=False,
    )

    predictions = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for batch, items in tqdm(loader, desc="Inference"):
            batch = copy_data_to_device(batch, device, non_blocking=True)
            output = model(batch)
            predictions.extend(
                postprocess_batch(output, batch, items, postprocessor)
            )

    predictions.sort(key=lambda item: item["annotation_id"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)
    print(f"Saved {len(predictions)} predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Full-model simple-query inference for RefSegRS/RRSIS-D"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
