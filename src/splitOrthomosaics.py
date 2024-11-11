# Script for splitting images in overlapping tiles

import cv2
import glob
import os
import numpy as np

from tqdm import tqdm, trange
from patchify import patchify
from pathlib import Path

HEIGHT = 2000
WIDTH = 2000
STEP = 1400
THRESHOLD = 0.90
BLACK = [0, 0, 0]

# Create directory for splitted images
Path("./Splitted").mkdir(exist_ok=True)

# Get orthomosaics
orthomosaics = glob.glob(r'/path/to/orthomosaics/*.tif')

for orthom in tqdm(orthomosaics, desc="Orthomosaics"):
    image = cv2.imread(orthom, cv2.IMREAD_UNCHANGED)

    # Configure final image size and `step` accordingly
    patches = patchify(image, (WIDTH, HEIGHT, 3), step=STEP)
    print(patches.shape)

    for i in trange(patches.shape[0], desc="Columns"):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]

            # Avoid all black patches
            if np.all(patch == 0):
                continue

            # Calculate image's area
            patch_area = patch.shape[0] * patch.shape[1]

            # Count number of black pixels
            black_pixels = np.count_nonzero(np.all(patch == BLACK, axis=2))
            percentage = black_pixels / patch_area

            # Select patches that have less than 90% of their area black to save
            if percentage < THRESHOLD:
                num = i * patches.shape[1] + j
                name = "./Splitted/" + file.replace('.tif', '') + f"_patch_{num:03}.jpg"

                cv2.imwrite(name, patch, [cv2.IMWRITE_JPEG_QUALITY, 75])

            del patch

    del image, patches

images = glob.glob("Splitted/*.jpg")

# print("Generated images:", len(images))
print("Final tally of images:", len(images))
