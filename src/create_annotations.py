from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import os
import json


def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

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
                    # sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))
                    sub_masks[pixel_str] = Image.new('1', (width + 0, height + 0)) # changed by mohsen

                # Set the pixel value to 1 (default is 0), accounting for padding
                # sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
                sub_masks[pixel_str].putpixel((x + 0, y + 0), 1) # changed by mohsen
    return sub_masks


def create_sub_mask_annotation(
        sub_mask):  # ref: https://github.com/akTwelve/cocosynth/blob/3909837290ab3511ff03ffe57ae870c929bd40a0/python/coco_json_utils.py#L151
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
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

            # if not poly.is_empty:
            #     polygons.append(poly)
            #     segmentation = np.array(poly.exterior.coords).ravel().tolist()
            #     segmentations.append(segmentation)
            if len(polygons) == 0:
                # This item doesn't have any visible polygons, ignore it
                # (This can happen if a randomly placed foreground is covered up
                #  by other foregrounds)
                continue

        #
        # polygons.append(poly)
        # segmentation = np.array(poly.exterior.coords).ravel().tolist()
        # segmentations.append(segmentation)

    return polygons, segmentations


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images


def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation


def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format
