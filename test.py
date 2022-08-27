import cv2
from inpaint import *
import matplotlib.pyplot as plt
import imageio


img = cv2.imread("imgs/1.jpg")
patchSize = 20 #size of the patch (without the overlap)
overlapSize = 5 #the width of the overlap region

img[40:100, 40:100, :] = 0
rect = np.zeros_like(img)
rect[40:100, 40:100, :] = 255

pbts = Inpaint(img, rect[:,:,0], patchSize, overlapSize, method=None, mirror_hor = True, mirror_vert = True)
inpaint = pbts.resolve()

images = [img, inpaint]
imageio.mimsave('assets/1.gif', images, duration=2.5)

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(inpaint)
plt.show()

img = cv2.imread("imgs/2.jpg")
patchSize = 20 #size of the patch (without the overlap)
overlapSize = 5 #the width of the overlap region

img[40:100, 60:100, :] = 0
rect = np.zeros_like(img)
rect[40:100, 60:100, :] = 255

pbts = Inpaint(img, rect[:,:,0], patchSize, overlapSize, mirror_hor = True, mirror_vert = True)
inpaint = pbts.resolve()

images = [img, inpaint]
imageio.mimsave('assets/2.gif', images, duration=2.5)

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(inpaint)
plt.show()

img = cv2.imread("imgs/3.jpg")
patchSize = 30 #size of the patch (without the overlap)
overlapSize = 10 #the width of the overlap region

img[40:100, 60:120, :] = 0
rect = np.zeros_like(img)
rect[40:100, 60:120, :] = 255

pbts = Inpaint(img, rect[:,:,0], patchSize, overlapSize, mirror_hor = True, mirror_vert = True)
inpaint = pbts.resolve()

images = [img, inpaint]
imageio.mimsave('assets/3.gif', images, duration=2.5)

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(inpaint)
plt.show()
img = cv2.imread("imgs/4.jpg")
patchSize = 110 #size of the patch (without the overlap)
overlapSize = 70 #the width of the overlap region

img[500:600, 500:1300, :] = 0
img[900:1000, 500:1300, :] = 0
rect = np.zeros_like(img)
rect[500:600, 500:1300, :] = 255
rect[900:1000, 500:1300, :] = 255

pbts = Inpaint(img, rect[:,:,0], patchSize, overlapSize, mirror_hor = True, mirror_vert = True, method=None)
inpaint = pbts.resolve()

images = [img, inpaint]
imageio.mimsave('assets/4.gif', images, duration=2.5)

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(inpaint)
plt.show()

