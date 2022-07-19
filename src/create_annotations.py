import glob
import os
import json
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import numpy as np
import cv2
from itertools import chain  # for flattening lists


class CreateCocoFormatInstances():
    def __init__(self):
        self.coco_instance = {}
        self.coco_instance["info"] = {}
        self.coco_instance["licenses"] = []
        self.coco_instance["images"] = []
        self.coco_instance["annotations"] = []
        self.coco_instance["categories"] = []

        self.coco_panoptic = {}
        self.coco_panoptic["info"] = {}
        self.coco_panoptic["licenses"] = []
        self.coco_panoptic["images"] = []
        self.coco_panoptic["annotations"] = []
        self.coco_panoptic["categories"] = []

        self.color_palette = {}
        self.cache_image_id = 0
        self.cache_annotation_id = 0
        self.cache_category_id = 0
        self.cache_segmentation_id = 0
        self.images_info = None  # to cache categories information

    def load_categories(self):
        with open('panoptic_coco_categories.json', 'r') as f:
            self.images_info = json.load(f)
        for cat in self.images_info['categories']:
            instance_category = {
                "supercategory": cat['supercategory'],  # no super category for now (replace with "if", later)
                "isthing": cat['isthing'],  # countable or not
                "id": cat['id'],
                "name": cat['name'],
                "color": cat['color']
            }
            panoptic_category = {
                "supercategory": cat['supercategory'],  # no super category for now (replace with "if", later)
                "isthing": cat['isthing'],  # countable or not
                "id": cat['id'],
                "name": cat['name'],
                "color": cat['color']
            }

            self.coco_instance["categories"].append(instance_category)
            self.coco_panoptic["categories"].append(panoptic_category)
            self.color_palette[cat['id']] = cat['color']

        self.color_palette_list = list(chain.from_iterable([self.color_palette[key] for key in self.color_palette]))

    def create_annotations(self, mask_image, imshow=False):
        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"

        self.cache_file_name = original_file_name

        # 1. Load mask image
        mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)
        # update the coco format with the image info
        image = {
            "file_name": original_file_name,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "id": self.cache_image_id
        }
        # print("----", image)
        self.coco_instance["images"].append(image)
        self.coco_panoptic["images"].append(image)  # Same as instance format

        if imshow:
            cv2.imshow(f"mask {original_file_name}", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # mask = Image.fromarray(mask) # PIL image
        # mask.putpalette(self.color_palette_list)

        # 2. Create sub-masks
        sub_masks = self.create_sub_masks(mask)

        # 3. Create annotations
        self.create_annotations_from_sub_masks(sub_masks)

    def create_sub_masks(self, mask):
        sub_masks = {}
        for sub_mask_id in np.unique(mask):
            sub_mask = mask == sub_mask_id
            sub_mask = sub_mask.astype(np.uint8)
            sub_masks[str(sub_mask_id)] = sub_mask
        return sub_masks


    def create_annotations_from_sub_masks(self, sub_masks):
        # 4. Create annotations
        for sub_mask_id, sub_mask in sub_masks.items():
            # print(sub_mask_id)

            self.cache_category_id = int(sub_mask_id) // self.images_info['info'][
                'categories_color_bin_size']  # semantic

            contours, hierarchy = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:

                instance_annotation = {
                    "iscrowd": 0,
                    "id": self.cache_annotation_id,
                    "image_id": self.cache_image_id,
                    "category_id": self.cache_category_id,
                    "bbox": cv2.boundingRect(contour),
                    "area": cv2.contourArea(contour),
                    "segmentation": [contour.flatten().tolist()]}

                # panoptic
                pan_segment_info = {}
                pan_segments_info = []
                pan_segment_id = 0

                for i in contours:
                    pan_segment_info = {
                        "id": pan_segment_id,
                        "category_id": self.cache_category_id,
                        "iscrowd": 0,
                        "bbox": cv2.boundingRect(contour),
                        "area": cv2.contourArea(contour),
                    }
                    pan_segment_id += 1  # TODO: check if it is correct, compare JSON
                    pan_segments_info.append(pan_segment_info)

                if instance_annotation['area'] > 0:
                    self.coco_instance["annotations"].append(instance_annotation)
                    self.coco_panoptic["annotations"].append({
                        "segments_info": pan_segments_info,
                        "file_name": self.cache_file_name[:-4] + '.png',
                        "image_id": self.cache_image_id,
                    })

                    self.cache_annotation_id += 1

                # print( int(sub_mask_id)//10, sub_mask_id )

            #
            #
            #
            #
            #
            # # contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")
            # polygons = []
            # segmentations = []
            #
            # for contour in contours:
            #     # Flip from (row, col) representation to (x, y) and subtract the padding pixel
            #     for i in range(len(contour)):
            #         row, col = contour[i]
            #         contour[i] = (col - 1, row - 1)
            #
            #     # Make a polygon and simplify it
            #     poly = Polygon(contour)
            #     poly = poly.simplify(1.0, preserve_topology=False)
            #
            #     print(poly)
            #     if poly.area > 5:  # Ignore tiny polygons
            #         if poly.geom_type == 'MultiPolygon':
            #             # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
            #             poly = poly.convex_hull
            #
            #         if poly.geom_type == 'Polygon':  # Ignore if still not a Polygon (could be a line or point)
            #             polygons.append(poly)
            #             segmentation = np.array(poly.exterior.coords).ravel().tolist()
            #             segmentations.append(segmentation)
            #
            #         if len(polygons) == 0:
            #             # This item doesn't have any visible polygons, ignore it
            #             # (This can happen if a randomly placed foreground is covered up by other foregrounds)
            #             continue
            #
            # # Check if we have classes that are a multipolygon
            # multipolygon_ids = [] # list of ids of polygons that are multipolygons
            # if cat in multipolygon_ids:
            #     print("cat is multipolygon", cat)
            #     # Combine the polygons to calculate the bounding box and area
            #     multi_poly = MultiPolygon(polygons)
            #
            #     min_w, min_h, max_w, max_h = multi_poly.bounds
            #     annotation = {
            #         "segmentation": segmentations,
            #         "area": multi_poly.area,
            #         "iscrowd": 0,
            #         "image_id": self.cache_image_id,
            #         "bbox": [min_w, min_h, (max_w - min_w), (max_h - min_h)],
            #
            #         "id": self.cache_annotation_id,
            #         "category_id": self.cache_category_id,
            #     }
            #
            #     self.coco_format["annotations"].append(annotation)
            #     self.cache_annotation_id += 1
            #
            # else:
            #     print(f"{cat} is not a multipolygon!")
            #
            #     for i in range(len(polygons)):
            #         # Cleaner to recalculate this variable
            #         segmentations = [np.array(polygons[i].exterior.coords).ravel().tolist()]
            #         min_w, min_h, max_w, max_h = polygons[i].bounds
            #
            #     annotation = {
            #         "segmentation": segmentations,
            #         "area": polygons[i].area,
            #         "iscrowd": 0,
            #         "image_id": self.cache_image_id,
            #         "bbox": [min_w, min_h, (max_w - min_w), (max_h - min_h)],
            #
            #         "id": self.cache_annotation_id,
            #         "category_id": self.cache_category_id,
            #     }
            #     self.coco_format["annotations"].append(annotation)
            #     self.cache_annotation_id += 1
            #
