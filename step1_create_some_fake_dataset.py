"""
Create some fake dataset for COCO format creation.
Mohsen Azimi (2022)
https://mohsen-azimi.github.io
"""

import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
from PIL import Image  # pip install pillow
from itertools import chain  # for flattening lists
import json
import os

# if not os.path.exists(DIR_PATH):
#     os.makedirs(DIR_PATH)


height = 480  # height of the image
width = 640  # width of the image
fake_dataset = {"train": 4, "val": 2, "test": 2}  # number of images per category
# RGB

color_palette = {}
for j in range(250):
    color_palette[j] = (0, 0, 0)  # pre-defined colors for pillow palette

with open('panoptic_coco_categories.json', 'r') as f:
    images_info = json.load(f)

seed = 0
bin = images_info['info']['categories_color_bin_size']  # bin here is the max number of instances/category

for subset, n_images in fake_dataset.items():
    for i in range(n_images):
        seed += 1
        np.random.seed(seed)  # to get same random numbers every time
        rand_i = np.random.randint(50, 200)  # random integer between 10 and 200
        mask = np.zeros((height, width), dtype="uint8")  # uint16 to save more color ids easily!

        # Categorys: 0 = null, 1 = sky, 2 = ground, 3 = circle, 4 = rectangle,...

        # Create some random shapes

        category = 0  # null

        category += 1  # sky
        mask[:100, :] = category * bin + 1  # sky: insthing=False
        color_palette[category * bin + 1] = (0, 0, 255)

        category += 1  # ground
        mask[100:, :] = category * bin + 1  # ground: insthing=False
        color_palette[(category * bin + 1)] = (0, 255, 0)

        category += 1  # circle
        # 3. circle: color index from 201 to 300
        mask = cv2.circle(mask, (rand_i + 150, rand_i + 150), radius=50,
                          color=category * bin + 1, thickness=-1)  # circle1 (insthing=True)
        color_palette[category * bin + 1] = (0, 255, 255)

        mask = cv2.circle(mask, (rand_i + 250, rand_i + 250), radius=60,
                          color=category * bin + 2, thickness=-1)  # circle2 (insthing=True)
        color_palette[(category * bin + 2)] = (0, 200, 200)

        category += 1  # rectangle
        mask = cv2.rectangle(mask, pt1=(rand_i, rand_i), pt2=(rand_i + 50, rand_i + 50),
                             color=category * bin + 1, thickness=-1)  # rectangle (insthing=True)
        color_palette[(category * bin + 1)] = (200, 200, 100)

        cv2.imwrite(f'coco_panoptic/panoptic_gray/{subset}/{subset}_{i}.png', mask)  # save the mask image

        mask = Image.fromarray(mask)  # PIL image

        mask.putpalette(list(chain.from_iterable([color_palette[key] for key in color_palette])))  # set color palette

        if mask.mode != 'RGB':
            mask = mask.convert('RGB')  # convert to RGB

        mask.save(f"coco_panoptic/{subset}2017/{subset}_{i}.jpg")  # assume images = colorized masks
        mask.save(f"coco_panoptic/panoptic_{subset}2017/{subset}_{i}.png")
    print(f"created {i + 1} {subset} images.")

    # TODO: create the update functions for the JSON file, images_info.json

if __name__ == "__main__":
    print("Done!")
