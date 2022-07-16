import cv2
import numpy

# use numpy to create an array of color
height = 480
width = 640

color1 = (255, 0, 0)  # b
color2 = (0, 255, 0)  # g
color3 = (0, 0, 255)  # r
pixel_array = numpy.full((height, width, 3), color1, dtype=numpy.uint8)

pixel_array[0:100, 0:100, :] = color2
pixel_array[100:200, 100:200, :] = color3

for i in range(5):
    cv2.imwrite(f'coco_panoptic/panoptic_train2017/fake_train_{i}.png', pixel_array)  # annotation
    cv2.imwrite(f'coco_panoptic/train2017/fake_train_{i}.jpg', pixel_array)  # data

    cv2.imwrite(f'coco_panoptic/panoptic_val2017/fake_val_{i}.png', pixel_array)  # annotation
    cv2.imwrite(f'coco_panoptic/val2017/fake_val_{i}.jpg', pixel_array)  # data

print(f"created {i} images")
# cv2.imshow('image', pixel_array)
# cv2.waitKey(0)
