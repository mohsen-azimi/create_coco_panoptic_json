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

print(pixel_array.shape)
for i in range(10):
    cv2.imwrite(f'dataset/annotations/train/fake_{i}.png', pixel_array)  # annotation
    cv2.imwrite(f'dataset/train/fake_{i}.jpg', pixel_array)  # data

    cv2.imwrite(f'dataset//annotations/val/fake_{i + 1000}.png', pixel_array)  # annotation
    cv2.imwrite(f'dataset/val/fake_{i + 1000}.jpg', pixel_array)  # data

# cv2.imshow('image', pixel_array)
# cv2.waitKey(0)
