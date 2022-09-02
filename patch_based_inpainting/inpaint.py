'''
MIT License
Copyright (c) 2022 Analyzable
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from math import floor, ceil
from random import randint

from sklearn.neighbors import KDTree
from skimage.util.shape import view_as_windows
from skimage.filters import gaussian
from skimage.transform import resize, rotate
from skimage import exposure


class Inpaint:
    '''
    The Inpaint object will performed patch-based inpainting.
    Usage: create the object with parameters, call object.resolve().

    Parameters
    ----------
    image : array
        The image to inpaint.
    mask : array
        The mask of the same size as the image, all value > 0 will be inpainted.
    patch_size : int
        The size of one square patch.
    overlap_size : int
        The size of the overlap between patch.
    training_area : tuple
        The rectangle (x, y, width, height) of the area for the training. If None will use all the image.
    window_step : int
        The shape of the elementary n-dimensional orthotope of the rolling window view. If None will be autocomputed. Can lead to a RAM saturation if to small.
    mirror_hor : bool
        Compute the horizontal mirror of each patch for training.
    mirror_vert : bool
        Compute the vertical mirror of each patch for training.
    rotation : list
        Compute the given rotations in degrees of each patch for training.
    method : str
        Method to use for blending adjacent patches. blend: feathering blending ; linear: mean blending ; gaussian: gaussian blur blending ; None: no blending.

    '''

    def __init__(self, image, mask, patch_size, overlap_size, training_area=None, window_step=None, mirror_hor=True, mirror_vert=True, rotation=None, method="blend"):

        self.image = np.float32(image)
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.training_area = training_area
        self.mirror_hor = mirror_hor
        self.mirror_vert = mirror_vert
        self.rotation = rotation
        self.total_patches_count = 0
        if window_step is None:
            self.window_step = np.max(np.shape(self.image))//30
        else:
            self.window_step = window_step
        self.method = method
        self.iter = 0

        self.rects = []
        self.mask = np.uint8(mask)
        self.compute_rect()

        self.example_patches = self.compute_patches()
        self.kdtree = self.init_KDtrees()

        self.PARM_truncation = 0.8
        self.PARM_attenuation = 2

        self.blending_mask = np.ones(
            (self.patch_size+2*self.overlap_size, self.patch_size+2*self.overlap_size, 3))
        self.blending_mask[0:self.overlap_size//3, :, :] = 0
        self.blending_mask[:, 0:self.overlap_size//3, :] = 0
        self.blending_mask[-self.overlap_size//3::, :, :] = 0
        self.blending_mask[:, -self.overlap_size//3::, :] = 0
        self.blending_mask = gaussian(
            self.blending_mask, sigma=self.overlap_size//2, preserve_range=True, channel_axis=2)
        self.blending_mask = exposure.rescale_intensity(self.blending_mask)

    def compute_rect(self):
        contours, __ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            rect = list(cv2.boundingRect(i))
            rect[2] = min(rect[2], self.image.shape[1] - rect[0])
            rect[3] = min(rect[3], self.image.shape[0] - rect[1])
            win = np.asarray(
                [(i+1)*self.patch_size + i*self.overlap_size for i in range(self.image.shape[0])])
            rect[2] = win[np.where((win >= rect[2]) == True)[0][0]]
            rect[3] = win[np.where((win >= rect[3]) == True)[0][0]]

            if rect[2] > self.image.shape[1] - rect[0]:
                rect[2] = win[np.where((win >= rect[2]) == True)[0][0] - 2]
            if rect[3] > self.image.shape[0] - rect[1]:
                rect[3] = win[np.where((win >= rect[3]) == True)[0][0] - 2]

            self.rects.append(rect)
            self.mask[rect[1]:rect[1] + rect[3],
                      rect[0]:rect[0] + rect[2]] = 255

    def compute_patches(self):

        kernel_size = self.patch_size + 2 * self.overlap_size
        self.image[self.mask > 0] = np.nan

        if self.training_area is not None:
            result = view_as_windows(self.image[self.training_area[1]: self.training_area[1] + self.training_area[3], self.training_area[0]
                                     : self.training_area[0] + self.training_area[2], :], [kernel_size, kernel_size, 3], self.window_step)
        else:
            result = view_as_windows(
                self.image, [kernel_size, kernel_size, 3], self.window_step)

        if self.rotation is not None:
            for i in self.rotation:
                if self.training_area is not None:
                    rot = rotate(self.image[self.training_area[1]: self.training_area[1] + self.training_area[3],
                                 self.training_area[0]: self.training_area[0] + self.training_area[2], :], i)
                else:
                    rot = rotate(self.image, i)
                result = np.concatenate((result, view_as_windows(
                    rot, [kernel_size, kernel_size, 3], self.window_step)))

        shape = np.shape(result)
        result = result.reshape(shape[0]*shape[1], kernel_size, kernel_size, 3)
        delete = []
        for i, j in enumerate(result):
            if np.isnan(np.sum(j)):
                delete.append(i)
        result = np.delete(result, delete, axis=0)

        shape = np.shape(result)
        self.total_patches_count = shape[0]

        if self.mirror_hor:
            hor_result = np.zeros(np.shape(result))

            for i in range(self.total_patches_count):
                hor_result[i] = result[i][::-1, :, :]
            result = np.concatenate((result, hor_result))

        if self.mirror_vert:
            vert_result = np.zeros((shape[0], kernel_size, kernel_size, 3))
            for i in range(self.total_patches_count):
                vert_result[i] = result[i][:, ::-1, :]
            result = np.concatenate((result, vert_result))
        return result

    def get_combined_overlap(self, overlaps):
        shape = np.shape(overlaps[0])
        if len(shape) > 1:
            combined = np.zeros((shape[0], shape[1]*len(overlaps)))
            for i, j in enumerate(overlaps):
                combined[0:shape[0], shape[1]*i:shape[1]*(i+1)] = j
        else:
            combined = np.zeros((shape[0]*len(overlaps)))
            for i, j in enumerate(overlaps):
                combined[shape[0]*i:shape[0]*(i+1)] = j
        return combined

    def init_KDtrees(self, leaf_size=25):
        top_overlap = self.example_patches[:, 0:self.overlap_size, :, :]
        bottom_overlap = self.example_patches[:, -self.overlap_size::, :, :]
        left_overlap = self.example_patches[:, :, 0:self.overlap_size, :]
        right_overlap = self.example_patches[:, :, -self.overlap_size::, :]

        shape_top = np.shape(top_overlap)
        shape_bottom = np.shape(bottom_overlap)
        shape_left = np.shape(left_overlap)
        shape_right = np.shape(right_overlap)

        flatten_top = top_overlap.reshape(shape_top[0], -1)
        flatten_bottom = bottom_overlap.reshape(shape_bottom[0], -1)
        flatten_left = left_overlap.reshape(shape_left[0], -1)
        flatten_right = right_overlap.reshape(shape_right[0], -1)

        flatten_combined_4 = self.get_combined_overlap(
            [flatten_top, flatten_bottom, flatten_left, flatten_right])
        flatten_combined_3 = self.get_combined_overlap(
            [flatten_top, flatten_left, flatten_right])
        flatten_combined_3_bis = self.get_combined_overlap(
            [flatten_top, flatten_bottom, flatten_left])
        flatten_combined_2 = self.get_combined_overlap(
            [flatten_top, flatten_left])
        flatten_combined_2_bis = self.get_combined_overlap(
            [flatten_top, flatten_bottom])
        flatten_combined_2_bis_1 = self.get_combined_overlap(
            [flatten_left, flatten_right])

        tree_top = KDTree(flatten_top, leaf_size=leaf_size)
        tree_left = KDTree(flatten_left, leaf_size=leaf_size)
        tree_combined_4 = KDTree(flatten_combined_4, leaf_size=leaf_size)
        tree_combined_3 = KDTree(flatten_combined_3, leaf_size=leaf_size)
        tree_combined_3_bis = KDTree(
            flatten_combined_3_bis, leaf_size=leaf_size)
        tree_combined_2 = KDTree(flatten_combined_2, leaf_size=leaf_size)
        tree_combined_2_bis = KDTree(
            flatten_combined_2_bis, leaf_size=leaf_size)
        tree_combined_2_bis_1 = KDTree(
            flatten_combined_2_bis_1, leaf_size=leaf_size)
        return {"t": tree_top, "l": tree_left, "tblr": tree_combined_4, "tlr": tree_combined_3, "tl": tree_combined_2, "tbl": tree_combined_3_bis, "tb": tree_combined_2_bis, "lr": tree_combined_2_bis_1}

    def find_most_similar_patches(self, overlap_top, overlap_bottom, overlap_left, overlap_right, k=5):
        if (overlap_top is not None) and (overlap_bottom is not None) and (overlap_left is not None) and (overlap_right is not None):
            combined = self.get_combined_overlap(
                [overlap_top.reshape(-1), overlap_bottom.reshape(-1), overlap_left.reshape(-1), overlap_right.reshape(-1)])
            dist, ind = self.kdtree["tblr"].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is None) and (overlap_left is not None) and (overlap_right is not None):
            combined = self.get_combined_overlap(
                [overlap_top.reshape(-1), overlap_left.reshape(-1), overlap_right.reshape(-1)])
            dist, ind = self.kdtree["tlr"].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is not None) and (overlap_left is not None) and (overlap_right is None):
            combined = self.get_combined_overlap(
                [overlap_top.reshape(-1), overlap_bottom.reshape(-1), overlap_left.reshape(-1)])
            dist, ind = self.kdtree["tbl"].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is None) and (overlap_left is not None) and (overlap_right is None):
            combined = self.get_combined_overlap(
                [overlap_top.reshape(-1), overlap_left.reshape(-1)])
            dist, ind = self.kdtree["tl"].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is None) and (overlap_left is None) and (overlap_right is None):
            dist, ind = self.kdtree["t"].query([overlap_top.reshape(-1)], k=k)
        elif (overlap_top is None) and (overlap_bottom is None) and (overlap_left is not None) and (overlap_right is None):
            dist, ind = self.kdtree["l"].query([overlap_left.reshape(-1)], k=k)
        elif (overlap_top is not None) and (overlap_bottom is not None) and (overlap_left is None) and (overlap_right is None):
            combined = self.get_combined_overlap(
                [overlap_top.reshape(-1), overlap_bottom.reshape(-1)])
            dist, ind = self.kdtree["tb"].query([combined], k=k)
        elif (overlap_top is None) and (overlap_bottom is None) and (overlap_left is not None) and (overlap_right is not None):
            combined = self.get_combined_overlap(
                [overlap_left.reshape(-1), overlap_right.reshape(-1)])
            dist, ind = self.kdtree["lr"].query([combined], k=k)
        elif (overlap_top is None) and (overlap_bottom is None) and (overlap_left is None) and (overlap_right is None):
            dist, ind = [None], [0]
        else:
            raise Exception(
                "ERROR: no valid overlap area is passed to -findMostSimilarPatch-")
        dist = dist[0]
        ind = ind[0]
        return dist, ind

    def resolve(self):
        '''
        Start the inpainting.
        '''

        for rect in self.rects:
            x0 = int(rect[0] - self.overlap_size)
            y0 = int(rect[1] - self.overlap_size)

            step_x = rect[2] // (self.patch_size+self.overlap_size) + 1
            step_y = rect[3] // (self.patch_size+self.overlap_size) + 1

            for i in range(step_y):  # Y
                for j in range(step_x):  # X
                    x = max(0, x0)
                    y = max(0, y0)

                    if j == step_x - 1:
                        overlap_right = self.image[y:y+self.patch_size+2*self.overlap_size, x +
                                                   self.patch_size+self.overlap_size:x+self.patch_size+2*self.overlap_size, :]
                    else:
                        overlap_right = None
                    if i == step_y - 1:
                        overlap_bottom = self.image[y + self.patch_size+self.overlap_size:y +
                                                    self.patch_size+2*self.overlap_size, x:x+self.patch_size+2*self.overlap_size, :]
                    else:
                        overlap_bottom = None

                    if y0 >= 0:
                        overlap_top = self.image[y:y+self.overlap_size,
                                                 x:x+self.patch_size+2*self.overlap_size, :]
                    else:
                        overlap_top = None

                    if x0 >= 0:
                        overlap_left = self.image[y:y+self.patch_size +
                                                  2*self.overlap_size, x:x+self.overlap_size, :]
                    else:
                        overlap_left = None

                    dist, ind = self.find_most_similar_patches(
                        overlap_top, overlap_bottom, overlap_left, overlap_right)

                    # TODO: check if is not a mirror and check precision of overlapping at right and bottom edges

                    if dist is not None:
                        probabilities = self.distances2probability(
                            dist, self.PARM_truncation, self.PARM_attenuation)
                        patch_id = np.random.choice(ind, 1, p=probabilities)
                    else:
                        patch_id = np.random.choice(
                            1, self.total_patches_count)

                    self.image[y:y+self.patch_size+2*self.overlap_size, x:x+self.patch_size+2*self.overlap_size, :] = self.merge(
                        self.image[y:y+self.patch_size+2*self.overlap_size, x:x+self.patch_size+2*self.overlap_size, :], self.example_patches[patch_id[0], :, :, :], method=self.method)
                    x0 += self.patch_size + self.overlap_size
                x0 = rect[0] - self.overlap_size
                y0 += self.patch_size + self.overlap_size
        return np.uint8(self.image)

    def merge(self, image_0, image_1, method="linear"):
        non_zeros = ~np.isnan(image_0)  # Overlap area
        zeros = np.isnan(image_0)  # patch_size area
        if method == "linear":
            image_0[zeros] = image_1[zeros]
            image_0[non_zeros] = (image_0[non_zeros] + image_1[non_zeros]) / 2
        elif method == "gaussian":
            image_0 = image_1
            image_0[non_zeros] = gaussian(
                image_0[non_zeros], sigma=1, preserve_range=True, channel_axis=2)
        elif method == "blend":
            image_0[zeros] = image_1[zeros]*self.blending_mask[zeros]
            image_0[non_zeros] = image_0[non_zeros] * \
                (1 - self.blending_mask[non_zeros]) + \
                image_1[non_zeros]*self.blending_mask[non_zeros]
        else:
            image_0 = image_1
        return image_0

    def distances2probability(self, distances, PARM_truncation, PARM_attenuation):

        probabilities = 1 - distances / np.max(distances)
        probabilities *= (probabilities > PARM_truncation)
        # attenuate the values
        probabilities = pow(probabilities, PARM_attenuation)
        # check if we didn't truncate everything!
        if np.sum(probabilities) == 0:
            # then just revert it
            probabilities = 1 - distances / np.max(distances)
            # truncate the values (we want top truncate%)
            probabilities *= (probabilities >
                              PARM_truncation*np.max(probabilities))
            probabilities = pow(probabilities, PARM_attenuation)
        # normalize so they add up to one
        probabilities /= np.sum(probabilities)

        return probabilities
