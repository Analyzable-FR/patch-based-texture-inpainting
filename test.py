import cv2
from inpaint import *
import matplotlib.pyplot as plt

img = cv2.imread("imgs/2.jpg")
patchSize = 20 #size of the patch (without the overlap)
overlapSize = 15 #the width of the overlap region

img[40:100, 60:100, :] = 0
rect=[60, 40, 40, 60]

pbts = Inpaint(img, rect, patchSize, overlapSize, window_step = 5, mirror_hor = True, mirror_vert = True)
inpaint = pbts.resolve()

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(inpaint)
plt.show()
