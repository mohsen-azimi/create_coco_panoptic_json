# Create Panoptic COCO Annotation files (*.JSON)


---

#   

## 1. Clone the repository

`$ git clone https://github.com/mohsen-azimi/smart-4wd-robot.git robot`

`$ cd robot`

## 2. Create `conda` environment

`$ conda create -n panoptic python=3.8 -y` 

`$ conda activate panoptic`

`$ pip install -r requirements.txt`

## 3. Step1. Create some fake data

`python step1_create_some_fake_dataset.py`

## 4. Create json files

`python step2_create_panoptic_json.py` 
`python step3_create_instances_json.py`