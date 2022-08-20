import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from math import floor, ceil
from random import randint

from sklearn.neighbors import KDTree
from skimage.util.shape import view_as_windows


class Inpaint:

    def __init__(self, image, rect, patch_size, overlap_size, window_step=5, mirror_hor = True, mirror_vert = True):

        self.image = np.float32(image)
        self.output = np.zeros_like(image)
        self.rect = rect
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.mirror_hor = mirror_hor
        self.mirror_vert = mirror_vert
        self.total_patches_count = 0
        self.window_step = window_step
        self.iter = 0

        self.example_patches = self.compute_patches()
        self.kdtree = self.init_KDtrees()

        self.PARM_truncation = 0.8
        self.PARM_attenuation = 2

        win = np.asarray([(i+1)*self.patch_size + i*self.overlap_size for i in range(20)])
        self.rect[2] = win[np.where((win>=self.rect[2])==True)[0][0]]
        self.rect[3] = win[np.where((win>=self.rect[3])==True)[0][0]]

    def compute_patches(self):

        kernel_size = self.patch_size + 2 * self.overlap_size
        self.image[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2], :] = np.nan
        result = view_as_windows(self.image, [kernel_size, kernel_size, 3] , self.window_step)


        shape = np.shape(result)
        result = result.reshape(shape[0]*shape[1], kernel_size, kernel_size, 3)
        delete = []
        for i, j in enumerate(result):
            if np.isnan(np.sum(j)):
                delete.append(i)
        result = np.delete(result, delete, axis=0)
        self.image[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2], :] = 0

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

    def get_combined_overlap(self, top, bottom, left, right):
        shape = np.shape(top)
        if bottom is None and right is None:
            if len(shape) > 1:
                combined = np.zeros((shape[0], shape[1]*2))
                combined[0:shape[0], 0:shape[1]] = top
                combined[0:shape[0], shape[1]:shape[1]*2] = left
            else:
                combined = np.zeros((shape[0]*2))
                combined[0:shape[0]] = top
                combined[shape[0]:shape[0]*2] = left
        elif bottom is None and right is not None :
            if len(shape) > 1:
                combined = np.zeros((shape[0], shape[1]*3))
                combined[0:shape[0], 0:shape[1]] = top
                combined[0:shape[0], shape[1]:shape[1]*2] = left
                combined[0:shape[0], shape[1]*2:shape[1]*3] = right
            else:
                combined = np.zeros((shape[0]*3))
                combined[0:shape[0]] = top
                combined[shape[0]:shape[0]*2] = left
                combined[shape[0]*2:shape[0]*3] = right
        elif right is None and bottom is not None :
            if len(shape) > 1:
                combined = np.zeros((shape[0], shape[1]*3))
                combined[0:shape[0], 0:shape[1]] = top
                combined[0:shape[0], shape[1]:shape[1]*2] = bottom
                combined[0:shape[0], shape[1]*2:shape[1]*3] = left
            else:
                combined = np.zeros((shape[0]*3))
                combined[0:shape[0]] = top
                combined[shape[0]:shape[0]*2] = bottom
                combined[shape[0]*2:shape[0]*3] = left
        else:
            if len(shape) > 1:
                combined = np.zeros((shape[0], shape[1]*4))
                combined[0:shape[0], 0:shape[1]] = top
                combined[0:shape[0], shape[1]:shape[1]*2] = bottom
                combined[0:shape[0], shape[1]*2:shape[1]*3] = left
                combined[0:shape[0], shape[1]*3:shape[1]*4] = right
            else:
                combined = np.zeros((shape[0]*4))
                combined[0:shape[0]] = top
                combined[shape[0]:shape[0]*2] = bottom
                combined[shape[0]*2:shape[0]*3] = left
                combined[shape[0]*3:shape[0]*4] = right
        return combined

    def init_KDtrees(self):
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

        flatten_combined_4 = self.get_combined_overlap(flatten_top, flatten_bottom, flatten_left, flatten_right)
        flatten_combined_3 = self.get_combined_overlap(flatten_top, None, flatten_left, flatten_right)
        flatten_combined_3_bis = self.get_combined_overlap(flatten_top, flatten_bottom, flatten_left, None)
        flatten_combined_2 = self.get_combined_overlap(flatten_top, None, flatten_left, None)

        tree_top = KDTree(flatten_top)
        tree_bottom = KDTree(flatten_bottom)
        tree_left = KDTree(flatten_left)
        tree_right = KDTree(flatten_right)
        tree_combined_4 = KDTree(flatten_combined_4)
        tree_combined_3 = KDTree(flatten_combined_3)
        tree_combined_3_bis = KDTree(flatten_combined_3_bis)
        tree_combined_2 = KDTree(flatten_combined_2)
        return [tree_top, tree_bottom, tree_left, tree_right, tree_combined_4, tree_combined_3, tree_combined_2, tree_combined_3_bis] # tree_combined

    def find_most_similar_patches(self, overlap_top, overlap_bottom, overlap_left, overlap_right, k=5):
        if (overlap_top is not None) and (overlap_bottom is not None) and (overlap_left is not None) and (overlap_right is not None):
            combined = self.get_combined_overlap(overlap_top.reshape(-1), overlap_bottom.reshape(-1), overlap_left.reshape(-1), overlap_right.reshape(-1))
            dist, ind = self.kdtree[4].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is None) and (overlap_left is not None) and (overlap_right is not None):
            combined = self.get_combined_overlap(overlap_top.reshape(-1), None, overlap_left.reshape(-1), overlap_right.reshape(-1))
            dist, ind = self.kdtree[5].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is not None) and (overlap_left is not None) and (overlap_right is None):
            combined = self.get_combined_overlap(overlap_top.reshape(-1), overlap_bottom.reshape(-1), overlap_left.reshape(-1), None)
            dist, ind = self.kdtree[7].query([combined], k=k)
        elif (overlap_top is not None) and (overlap_bottom is None) and (overlap_left is not None) and (overlap_right is None):
            combined = self.get_combined_overlap(overlap_top.reshape(-1), None, overlap_left.reshape(-1), None)
            dist, ind = self.kdtree[6].query([combined], k=k)
        else:
            raise Exception("ERROR: no valid overlap area is passed to -findMostSimilarPatch-")
        dist = dist[0]
        ind = ind[0]
        return dist, ind

    def idCoordTo2DCoord(self, idCoord, imgSize):
        row = int(floor(idCoord / imgSize[0]))
        col = int(idCoord - row * imgSize[1])
        return [row, col]

    def resolve(self):

        x0 = int(self.rect[0] - self.overlap_size)
        y0 = int(self.rect[1] - self.overlap_size)

        step_x = self.rect[2] // (self.patch_size+self.overlap_size) + 1
        step_y = self.rect[3] // (self.patch_size+self.overlap_size) + 1

        for i in range(step_y): # Y
            for j in range(step_x): # X
                if j == step_x - 1:
                    overlap_right = self.image[y0:y0+self.patch_size+2*self.overlap_size, x0 + self.patch_size+self.overlap_size:x0+self.patch_size+2*self.overlap_size, :]
                else:
                    overlap_right = None
                if i == step_y - 1:
                    overlap_bottom = self.image[y0 + self.patch_size+self.overlap_size:y0 + self.patch_size+2*self.overlap_size, x0:x0+self.patch_size+2*self.overlap_size, :]
                else:
                    overlap_bottom = None

                overlap_top = self.image[y0:y0+self.overlap_size, x0:x0+self.patch_size+2*self.overlap_size, :]
                overlap_left = self.image[y0:y0+self.patch_size+2*self.overlap_size, x0:x0+self.overlap_size, :]

                dist, ind = self.find_most_similar_patches(overlap_top, overlap_bottom, overlap_left, overlap_right)

                # TODO: check if is not a mirror and check precision of overlapping at right and bottom edges

                probabilities = self.distances2probability(dist, self.PARM_truncation, self.PARM_attenuation)
                patch_id = np.random.choice(ind, 1, p=probabilities)

                self.output[y0:y0+self.patch_size+2*self.overlap_size, x0:x0+self.patch_size+2*self.overlap_size, :] = self.example_patches[patch_id[0],:,:,:]
                x0 += self.patch_size + self.overlap_size
            x0 = self.rect[0] - self.overlap_size
            y0 += self.patch_size + self.overlap_size
        self.image[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2], :] = self.output[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2], :]
        return np.uint8(self.image)


    def distances2probability(self, distances, PARM_truncation, PARM_attenuation):

        probabilities = 1 - distances / np.max(distances)
        probabilities *= (probabilities > PARM_truncation)
        probabilities = pow(probabilities, PARM_attenuation) #attenuate the values
        #check if we didn't truncate everything!
        if np.sum(probabilities) == 0:
            #then just revert it
            probabilities = 1 - distances / np.max(distances)
            probabilities *= (probabilities > PARM_truncation*np.max(probabilities)) # truncate the values (we want top truncate%)
            probabilities = pow(probabilities, PARM_attenuation)
        probabilities /= np.sum(probabilities) #normalize so they add up to one

        return probabilities

