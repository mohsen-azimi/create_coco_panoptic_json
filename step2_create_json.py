import glob
import json
from src import CreateCocoFormatInstances as COCO


subsets = ["train", "val", "test"]


for subset in subsets:
    ##############################################################################
    my_coco = COCO()  # create an instance of the class CreateCocoFormat()
    my_coco.load_categories()  # load categories from images_info.json

    for mask_image in glob.glob(f"coco_panoptic/panoptic_{subset}2017/*.png"):
        my_coco.create_annotations(mask_image, imshow=False)  # create annotations from masks
        my_coco.cache_image_id += 1


    ##############################################################################
    # save the coco format
    with open(f"coco_panoptic/annotations/instances_{subset}2017.json", 'w') as f:
        json.dump(my_coco.coco_instance, f)
    with open(f"coco_panoptic/annotations/panoptic_{subset}2017.json", 'w') as f:
        json.dump(my_coco.coco_panoptic, f)

        print("Created %d annotations for '%s' images in folder" % (my_coco.cache_image_id, subset))
        if my_coco.cache_image_id == 0:
            print("No annotations created!")



