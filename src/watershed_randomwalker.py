from matplotlib.patches import Circle
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from scipy import ndimage




def segment_watershed_randomwalker(self, mask):
    distance = ndimage.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=mask)


def segment_random_walker(self, mask):
    distance = ndimage.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
    markers = measure.label(local_maxi)
    markers[~mask] = -1
    labels_rw = random_walker(mask, markers)




""" 
Try with Colab:

# Generate an initial image with two overlapping circles
x, y = np.indices((800, 900))

x1, y1, x2, y2, x3, y3 = 200, 200, 400, 400, 600, 660
r1, r2, r3 = 100, 100, 50
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3 ** 2

image = np.logical_or(mask_circle1, mask_circle2)
image = np.logical_or(image, mask_circle3)


# image = mask
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance
# to the background
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(
    distance, indices=False, footprint=np.ones((3, 3)), labels=image)
markers = measure.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)

markers[~image] = -1
labels_rw = random_walker(image, markers)

plt.figure(figsize=(12, 3.5))
plt.subplot(141)
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('image')
plt.subplot(142)
plt.imshow(-distance, interpolation='nearest')
plt.axis('off')
plt.title('distance map')
plt.subplot(143)
plt.imshow(labels_ws, cmap='nipy_spectral', interpolation='nearest')
plt.axis('off')
plt.title('watershed segmentation')
plt.subplot(144)
plt.imshow(labels_rw, cmap='nipy_spectral', interpolation='nearest')
plt.axis('off')
plt.title('random walker segmentation')

plt.tight_layout()
plt.show()
"""