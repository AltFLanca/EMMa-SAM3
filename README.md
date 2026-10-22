Inference run with command:

```
uv run src/inference.py --dataset-root /home/user/datasets/EMMa --asins /home/user/datasets/EMMa-ASINs/augmented_asins.npy --output /home/user/datasets/EMMa-SAM3/masks.h5 --sam3-root /home/user/models/facebook/sam3 --segmentation-threshold 0.25 --mask-threshold 0.5 --image-size 560 --thresholding-threshold 253 --thresholding-kernel-size 3 --thresholding-open-iterations 2
```

Then:

```
uv run src/valid_output_filtering.py --dataset-root /home/user/datasets/EMMa --augmentation-file /home/user/datasets/EMMa-SAM3/masks.h5
```