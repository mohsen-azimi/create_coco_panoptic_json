import glob
import os
import json
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import numpy as np

# from create_annotations import *
# from src.coco_viewer import CocoDataset

# from src.coco_to_yolo import COCO2YOLO

# mask_path = "dataset/Material Detection/mask_512/"
mask_path = "dataset/fake_mask/"
# Define the ids that are a multiplolygon. for example: wall, roof and sky
multipolygon_ids = []  # , 1, 2,3]

# =============================
# Step 1: Get the coco_panoptic_json_format
# =============================
# Get the standard COCO Panoptic JSON format
# coco_format = get_coco_panoptic_json_format()

coco_format = {
    "info": {},
    "licenses": [],
    "images": [{}],
    "categories": [{}],
    "annotations": [{}]
}

# =============================
# Step 2: get he category info
# =============================

category_ids = {
    # "obj00": 0,
    # "obj11": 1,
    # "obj22": 2,
    # "obj33": 3,
    "(255, 0, 0)": 25511,
    "(0, 255, 0)": 12551,
    "(0, 0, 255)": 11255,
    # "(255, 255, 255)": 255,
}

category_colors = {
    # "(0, 0, 0)": 0,  # null
    "(255, 0, 0)": 25511,
    "(0, 255, 0)": 12551,
    "(0, 0, 255)": 11255,
    # "(255, 255, 255)": 255,
    # "(0, 128, 0)": 2,
    # "(0, 0, 128)": 3,
    # "(128, 0, 0)": 4,
}
# Create category section
# coco_format["categories"] = create_category_annotation(category_ids)

coco_format["categories"] = []
for key, value in category_ids.items():
    category = {
        "supercategory": key,  # no super category for now (replace with "if", later)
        "isthing": 0,  # countable or not
        "id": value,
        "name": key

    }
    coco_format["categories"].append(category)

# print(coco_format) # check the output!
# =============================
# Step 3: create annotations
# =============================


# This id will be automatically increased as we go
# annotation_id = 0

coco_format["images"] = []
coco_format["annotations"] = []

image_id = 0
for mask_image in glob.glob(mask_path + "*.png"):
    # The mask image is *.png but the original image is *.jpg.
    # We make a reference to the original file in the COCO JSON file

    original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpeg"
    # print(original_file_name)
    # Open the image and (to be sure) we convert it to RGB
    mask_image_open = Image.open(mask_image).convert("RGB")
    width, height = mask_image_open.size
    # print(mask_image_open.size)
    # 1
    coco_format["images"].append({
            "file_name": original_file_name,
            "height": height,
            "width": width,
            "id": image_id
        })



    # 2. Sub masks
    sub_masks = {}
    for w in range(width):
        for h in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image_open.getpixel((w, h))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((w + 1, h + 1), 1)



    # 3. color & annotation
    segments_info = []  # reset for each file
    segment_id = 0
    for color, sub_mask in sub_masks.items():
        print(color)
        category_id = category_colors[color]

        # create_sub_mask_annotation
        contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y) and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)

            if poly.area > 5:  # Ignore tiny polygons
                if poly.geom_type == 'MultiPolygon':
                    # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                    poly = poly.convex_hull

                if poly.geom_type == 'Polygon':  # Ignore if still not a Polygon (could be a line or point)
                    polygons.append(poly)
                    # segmentation = np.array(poly.exterior.coords).ravel().tolist()
                    # segmentations.append(segmentation)

                if len(polygons) == 0:
                    # This item doesn't have any visible polygons, ignore it
                    # (This can happen if a randomly placed foreground is covered up by other foregrounds)
                    continue

        # Check if we have classes that are a multipolygon
        if category_id in multipolygon_ids:
            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)

            min_w, min_h, max_w, max_h = multi_poly.bounds
            segment_info = {
                "id": segment_id,
                "category_id": category_id,
                "iscrowd": 0,
                "bbox": [min_w, min_h, max_w - min_w, max_h - min_h],
                "area": multi_poly.area,
            }
            segments_info.append(segment_info)
            # segment_id += 1
        else:
            for i in range(len(polygons)):

                min_w, min_h, max_w, max_h = polygons[i].bounds
                print(polygons[i].bounds)
                segment_info = {
                    "id": segment_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "bbox": [int(min_w), int(min_h), int(max_w - min_w), int(max_h - min_h)],
                    # "segments_info": segmentations,
                    "area": polygons[i].area,
                    # "image_id": image_id,
                }

                segments_info.append(segment_info)
            # segment_id += 1
        segment_id += 1

    coco_format["annotations"].append({
                        "segments_info": segments_info,
                        "file_name": original_file_name[:-5] + '.png',
                        "image_id": width,
        })

    image_id += 1


# ===========================
# Final Step: Write to JSON
# ===========================
with open("output/panoptic_fake_mask.json", "w") as outfile:
    json.dump(coco_format, outfile)
    # print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))
    print("Done!")

