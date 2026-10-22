from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import hdf5plugin
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Sam3Config, Sam3Model, Sam3Processor

from emma import EmmaDataset
from emma.augment import HDF5Source
from emma.dataloader import EmmaMapDataset

import utils.logging_ as logging


SEGMENTATION_THRESHOLD = 0.25
MASK_THRESHOLD = 0.5
IMAGE_SIZE = 560
THRESHOLDING_THRESHOLD = 253
THRESHOLDING_KERNEL_SIZE = 3
THRESHOLDING_OPEN_ITERATIONS = 2
SEGMENTATION_COUNT_KEYS = ("title", "category", "thresholding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 foreground segmentation with a thresholding fallback over the EMMa dataset. "
            "Stores one finalized mask per object in an HDF5 augmentation file."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the EMMa dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 file. Defaults to <dataset_root>/<augmentation-name>.h5.",
    )
    parser.add_argument(
        "--augmentation-name",
        default="sam3_thresholding_masks",
        help="Registered augmentation name.",
    )
    parser.add_argument(
        "--asins",
        type=Path,
        required=True,
        help="Path to a .npy file containing the ASINs to process.",
    )
    parser.add_argument(
        "--sam3-root",
        type=Path,
        required=True,
        help="Path to the SAM3 model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="DataLoader and SAM3 inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor used only when num_workers > 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N pending ASINs.",
    )
    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=SEGMENTATION_THRESHOLD,
        help="SAM3 instance score threshold.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=MASK_THRESHOLD,
        help="Mask binarization threshold.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Square image size used by the SAM3 processor.",
    )
    parser.add_argument(
        "--thresholding-threshold",
        type=int,
        default=THRESHOLDING_THRESHOLD,
        help="Grayscale inverse-binary threshold for the thresholding fallback.",
    )
    parser.add_argument(
        "--thresholding-kernel-size",
        type=int,
        default=THRESHOLDING_KERNEL_SIZE,
        help="Morphological opening kernel size for the thresholding fallback.",
    )
    parser.add_argument(
        "--thresholding-open-iterations",
        type=int,
        default=THRESHOLDING_OPEN_ITERATIONS,
        help="Morphological opening iterations for the thresholding fallback.",
    )
    return parser.parse_args()


def load_asins(path: Path) -> list[str]:
    asins = np.load(path, allow_pickle=False)
    return [str(asin) for asin in asins.reshape(-1)]


def make_transform(ds: EmmaDataset):
    def transform(asin: str, rec: dict) -> dict:
        image_dict = ds.get_images(rec)
        first_img_key, first_img_bytes = next(iter(image_dict.items()))
        first_img = Image.open(io.BytesIO(first_img_bytes)).convert("RGB")

        return {
            "image": np.array(first_img),
            "title": ds.get_title(rec=rec),
            "categories": ds.get_category_names(rec=rec),
            "asin": asin,
            "image_key": first_img_key,
        }

    return transform


def collate_samples(samples: list[dict]) -> list[dict]:
    return samples


def make_writer(name: str, path: Path) -> HDF5Source:
    compression = hdf5plugin.Blosc(
        cname="lz4hc",
        clevel=5,
        shuffle=hdf5plugin.Blosc.SHUFFLE,
    )
    writer_kwargs: dict[str, Any] = {
        "mode": "a",
        "compression": compression,
        "shuffle": False,
        "fletcher32": False,
    }
    return HDF5Source(name, path, **writer_kwargs)


def counts_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".counts.json")


def load_segmentation_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {key: 0 for key in SEGMENTATION_COUNT_KEYS}

    raw_counts = json.loads(path.read_text())
    return {key: int(raw_counts.get(key, 0)) for key in SEGMENTATION_COUNT_KEYS}


def save_segmentation_counts(path: Path, counts: dict[str, int]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def missing_asins(writer: HDF5Source, asins: list[str]) -> list[str]:
    existing = set(writer.keys())
    return [asin for asin in asins if asin not in existing]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def most_specific_category(categories: list[Any]) -> str:
    if not categories:
        return ""

    category = categories[-1]
    while isinstance(category, (list, tuple)):
        if not category:
            return ""
        category = category[-1]

    return str(category).strip()


def image_to_rgb_array(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def title_segmentation_prompt(
    processor: Sam3Processor,
    device: torch.device,
    images: list[np.ndarray],
    titles: list[str],
    categories_batch: list[list[Any]],
):
    segmentation_inputs = processor(
        images=images,
        return_tensors="pt",
    )
    text_inputs = processor.tokenizer(
        titles,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    segmentation_inputs.update(text_inputs)
    return segmentation_inputs.to(device)


def category_segmentation_prompt(
    processor: Sam3Processor,
    device: torch.device,
    images: list[np.ndarray],
    titles: list[str],
    categories_batch: list[list[Any]],
):
    prompts = [most_specific_category(categories) for categories in categories_batch]
    return processor(
        images=images,
        text=prompts,
        return_tensors="pt",
    ).to(device)


PromptFn = Callable[
    [Sam3Processor, torch.device, list[np.ndarray], list[str], list[list[Any]]],
    Any,
]

SEGMENTATION_PROMPT_PIPELINE: list[dict[str, Any]] = [
    {
        "name": "truncated title",
        "count_key": "title",
        "prompt_fn": title_segmentation_prompt,
    },
    {
        "name": "last category",
        "count_key": "category",
        "prompt_fn": category_segmentation_prompt,
    },
]


def run_segmentation_prompt(
    model: Sam3Model,
    processor: Sam3Processor,
    device: torch.device,
    images: list[np.ndarray],
    titles: list[str],
    categories_batch: list[list[Any]],
    prompt_fn: PromptFn,
    segmentation_threshold: float,
    mask_threshold: float,
) -> list[dict[str, Any]]:
    segmentation_inputs = prompt_fn(processor, device, images, titles, categories_batch)

    with torch.inference_mode():
        segmentation_outputs = model(**segmentation_inputs)

    return processor.post_process_instance_segmentation(
        segmentation_outputs,
        threshold=segmentation_threshold,
        mask_threshold=mask_threshold,
        target_sizes=segmentation_inputs["original_sizes"].tolist(),
    )


def has_detection(
    segmentation_result: dict[str, Any],
    segmentation_threshold: float,
) -> bool:
    scores = segmentation_result["scores"]
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()

    return len(scores) > 0 and np.any(np.asarray(scores) >= segmentation_threshold)


def result_masks(segmentation_result: dict[str, Any]) -> np.ndarray:
    masks = segmentation_result["masks"]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()

    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    return masks


def segmentation_result_to_mask(
    image: np.ndarray,
    segmentation_result: dict[str, Any],
    mask_threshold: float,
) -> np.ndarray:
    image = image_to_rgb_array(image)
    masks = result_masks(segmentation_result)
    if len(masks) > 0:
        mask = np.any(masks > mask_threshold, axis=0)
    else:
        mask = np.zeros(image.shape[:2], dtype=bool)

    return mask.astype(np.uint8, copy=False)


def thresholding_segmentation(
    image_rgb: np.ndarray,
    threshold: int,
    kernel_size: int,
    open_iterations: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bin_img = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iterations,
    )
    return (bin_img > 127).astype(np.uint8, copy=False)


def segment_batch(
    model: Sam3Model,
    processor: Sam3Processor,
    device: torch.device,
    images: list[np.ndarray],
    titles: list[str],
    categories_batch: list[list[Any]],
    segmentation_threshold: float,
    mask_threshold: float,
    thresholding_threshold: int,
    thresholding_kernel_size: int,
    thresholding_open_iterations: int,
) -> list[tuple[np.ndarray, str]]:
    masks: list[np.ndarray | None] = [None] * len(images)
    count_keys: list[str | None] = [None] * len(images)
    unresolved = list(range(len(images)))

    for prompt in SEGMENTATION_PROMPT_PIPELINE:
        if not unresolved:
            break

        prompt_images = [images[i] for i in unresolved]
        prompt_titles = [titles[i] for i in unresolved]
        prompt_categories = [categories_batch[i] for i in unresolved]

        segmentation_results = run_segmentation_prompt(
            model=model,
            processor=processor,
            device=device,
            images=prompt_images,
            titles=prompt_titles,
            categories_batch=prompt_categories,
            prompt_fn=prompt["prompt_fn"],
            segmentation_threshold=segmentation_threshold,
            mask_threshold=mask_threshold,
        )

        next_unresolved = []
        for sample_idx, segmentation_result in zip(unresolved, segmentation_results):
            if not has_detection(segmentation_result, segmentation_threshold):
                next_unresolved.append(sample_idx)
                continue

            masks[sample_idx] = segmentation_result_to_mask(
                images[sample_idx],
                segmentation_result,
                mask_threshold=mask_threshold,
            )
            count_keys[sample_idx] = prompt["count_key"]

        unresolved = next_unresolved

    for sample_idx in unresolved:
        masks[sample_idx] = thresholding_segmentation(
            image_to_rgb_array(images[sample_idx]),
            threshold=thresholding_threshold,
            kernel_size=thresholding_kernel_size,
            open_iterations=thresholding_open_iterations,
        )
        count_keys[sample_idx] = "thresholding"

    finalized_results: list[tuple[np.ndarray, str]] = []
    for mask, count_key in zip(masks, count_keys):
        if mask is None or count_key is None:
            raise RuntimeError("Internal error: sample was not assigned a segmentation mask.")
        finalized_results.append((mask, count_key))

    return finalized_results


def main() -> None:
    args = parse_args()
    if not 0 <= args.thresholding_threshold <= 255:
        raise ValueError("--thresholding-threshold must be between 0 and 255.")
    if args.thresholding_kernel_size <= 0:
        raise ValueError("--thresholding-kernel-size must be positive.")
    if args.thresholding_open_iterations < 0:
        raise ValueError("--thresholding-open-iterations must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")

    dataset_root = args.dataset_root
    output_path = args.output or (dataset_root / f"{args.augmentation_name}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving augmentation to {output_path}")

    device = select_device(args.device)
    print(f"Loading SAM3 from {args.sam3_root} on {device} at {args.image_size}px")
    config = Sam3Config.from_pretrained(args.sam3_root)
    config.image_size = args.image_size

    model_kwargs: dict[str, Any] = {
        "config": config,
        "use_safetensors": True,
    }
    if device.type == "cuda":
        model_kwargs["device_map"] = "cuda"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = Sam3Model.from_pretrained(args.sam3_root, **model_kwargs).eval()
    if device.type != "cuda":
        model = model.to(device)

    processor = Sam3Processor.from_pretrained(
        args.sam3_root,
        size={"height": args.image_size, "width": args.image_size},
    )

    ds = EmmaDataset(dataset_root)
    asins = load_asins(args.asins)
    print(f"Candidate ASINs: {len(asins)}")

    writer = make_writer(args.augmentation_name, output_path)
    counts_path = counts_path_for(output_path)
    counts_file_exists = counts_path.exists()
    segmentation_counts = load_segmentation_counts(counts_path)
    print(f"Saving segmentation counts to {counts_path}")

    pending_asins = missing_asins(writer, asins)
    if not counts_file_exists and len(pending_asins) < len(asins):
        print("No counts file found; existing HDF5 masks cannot be classified, so counts start with newly saved masks.")
    if args.limit is not None:
        pending_asins = pending_asins[: args.limit]
    print(f"Pending ASINs: {len(pending_asins)}")
    if not pending_asins:
        writer.close()
        print("Nothing to do.")
        return

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_samples,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(
        EmmaMapDataset(ds, pending_asins, transform=make_transform(ds)),
        **loader_kwargs,
    )

    start = time.time()
    for batch_idx, batch in enumerate(loader):
        segmentation_results = segment_batch(
            model=model,
            processor=processor,
            device=device,
            images=[sample["image"] for sample in batch],
            titles=[sample["title"] for sample in batch],
            categories_batch=[sample["categories"] for sample in batch],
            segmentation_threshold=args.segmentation_threshold,
            mask_threshold=args.mask_threshold,
            thresholding_threshold=args.thresholding_threshold,
            thresholding_kernel_size=args.thresholding_kernel_size,
            thresholding_open_iterations=args.thresholding_open_iterations,
        )

        for sample, (mask, count_key) in zip(batch, segmentation_results):
            writer.save(sample["asin"], mask)
            segmentation_counts[count_key] += 1
            save_segmentation_counts(counts_path, segmentation_counts)

        logging.progress_bar(batch_idx, len(loader), 1, 1)

    writer.close()
    print(f"Elapsed: {time.time() - start:.2f}s")
    print("Done")


if __name__ == "__main__":
    main()
