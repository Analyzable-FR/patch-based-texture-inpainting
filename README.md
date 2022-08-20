# patch-based-texture-inpainting

Based on "Image Quilting for Texture Synthesis and Transfer" and "Real-Time Texture Synthesis by Patch-Based Sampling" papers and from the implementation of nopara https://github.com/anopara/patch-based-texture-synthesis.

## Usage
* image: the image to inpaint
* rect: (x0, y0, width, height) the area to inpaint
* patch_size: the size of patches that will be used to inpaint
* overlap_size: the size of the area to find similarity

## Example
![alt text](assets/2.gif)
![alt text](assets/3.gif)



## To Do 
* Clean the implementation
* Finish the implementation (see TO DO in code).
* Improve the algorithm at the border of the image.
* Take a list of rect
* Take a rect area for the training to speed-up for large images
