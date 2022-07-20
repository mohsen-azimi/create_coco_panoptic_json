# Create Panoptic COCO Annotation files (*.JSON)


---

#   

## 1. Clone the repository

`$ git clone https://github.com/mohsen-azimi/create_coco_panoptic_json.git robot`

`$ cd create_coco_panoptic_json`

## 2. Create `conda` environment

`$ conda create -n panoptic python=3.8 -y` 

`$ conda activate panoptic`

`$ pip install -r requirements.txt`

## 3. Create some fake data

`python step1_create_some_fake_dataset.py`

## 4. Create json files

`python step2_create_json.py` 

