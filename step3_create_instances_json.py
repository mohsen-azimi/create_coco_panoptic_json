import glob
import os
import json
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import numpy as np


for subset in ["train", "val"]:
    mask_path = f"coco_panoptic/panoptic_{subset}2017/"
    multipolygon_ids = []  # , 1, 2,3]


    # =============================
    # Step 1: Get the coco_panoptic_json_format
    # =============================
    # Get the standard COCO Panoptic JSON format

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
    coco_format["categories"] = []
    for key, value in category_ids.items():
        category = {
            "supercategory": key,  # no super category for now (replace with "if", later)
            # "isthing": 0,  # countable or not
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

    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    for mask_image in glob.glob(mask_path + "*.png"):

        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file

        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"
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
        # Initialize a dictionary of sub-masks indexed by RGB colors
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
        # segments_info = []  # reset for each file
        # segment_id = 0
        for color, sub_mask in sub_masks.items():
            # print(color)
            category_id = category_colors[color]

            # create_sub_mask_annotation
            contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")
            polygons = []
            segmentations = []

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
                        segmentation = np.array(poly.exterior.coords).ravel().tolist()
                        segmentations.append(segmentation)

                    if len(polygons) == 0:
                        # This item doesn't have any visible polygons, ignore it
                        # (This can happen if a randomly placed foreground is covered up by other foregrounds)
                        continue

            # Check if we have classes that are a multipolygon
            if category_id in multipolygon_ids:
                # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)

                min_w, min_h, max_w, max_h = multi_poly.bounds
                annotation = {
                    "segmentation": segmentation,
                    "area": multi_poly.area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [min_w, min_h, (max_w - min_w), (max_h - min_h)],

                    "id": annotation_id,
                    "category_id": category_id,
                }
                annotations.append(annotation)
                annotation_id += 1

            else:
                for i in range(len(polygons)):

                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                    min_w, min_h, max_w, max_h = polygons[i].bounds
                annotation = {
                    "segmentation": segmentation,
                    "area": polygons[i].area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [min_w, min_h, (max_w - min_w), (max_h - min_h)],

                    "id": annotation_id,
                    "category_id": category_id,
                }
                annotations.append(annotation)
                annotation_id += 1
        image_id += 1


    # ===========================
    # Final Step: Write to JSON
    # ===========================
    with open(f"coco_panoptic/annotations/instances_{subset}2017.json", 'w') as outfile:
        json.dump(coco_format, outfile)
        print("Created %d annotations for '%s' images in folder: %s" % (image_id, subset, mask_path))
        if image_id == 0:
            print("No annotations created for images in folder: %s" % mask_path)



