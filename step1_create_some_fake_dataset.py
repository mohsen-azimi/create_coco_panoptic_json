import cv2
import numpy
import numpy as np
# use numpy to create an array of color
height = 480
width = 640


for i in range(6):
    rand_i = np.random.randint(10, 300)

    bg = (0, 0, 0)  # b
    color1 = (255, 255, 255)
    # color2 = (255, 255, 255)
    pixel_array = numpy.full((height, width, 3), bg, dtype=numpy.uint8)

    pixel_array[rand_i:rand_i+100, rand_i:rand_i+100, :] = color1
    # pixel_array[110:200, 110:200, :] = color3



    cv2.imwrite(f'coco_panoptic/panoptic_train2017/fake_train_{i}.png', pixel_array)  # annotation
    cv2.imwrite(f'coco_panoptic/train2017/fake_train_{i}.jpg', pixel_array)  # data

    cv2.imwrite(f'coco_panoptic/panoptic_val2017/fake_val_{i}.png', pixel_array)  # annotation
    cv2.imwrite(f'coco_panoptic/val2017/fake_val_{i}.jpg', pixel_array)  # data

print(f"created {i+1} images")
