

## SOURCE >> https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html
#conda activate pytorch_venv
# pip install scikit-image
# cd /wm_img/code_modules/skimage_all
from skimage.io import imread  
from skimage.transform import rescale,resize,downscale_local_mean
import matplotlib.pyplot as plt


box_img = imread("../../inPut_dir/img/box.png")
#dog_img = imread("../../inPut_dir/img/dog.jpg")
dog_img = imread("../../inPut_dir/img/img_1.png")
print(type(box_img)) #<class 'numpy.ndarray'>
print("--box_img.shape---",box_img.shape)
print("--box_img.ndim---",box_img.ndim) # 2D
print("--box_img.size---",box_img.size) # The number of pixels - SIZE --box_img.size--- 72252
#
print("--dog_img.shape---",dog_img.shape)
print("--dog_img.ndim---",dog_img.ndim) # 3D
print("--dog_img.size---",dog_img.size) # The number of pixels - SIZE --dog_img.size--- 2099160
#
red_dog_img = dog_img[:, :, 0]
green_dog_img = dog_img[:, :, 1]
blue_dog_img = dog_img[:, :, 2]
#compare(dog_img, red_dog_img, "Red Channel of the Image", cmap_type="Reds_r")
#
plt.imshow(red_dog_img ,cmap="Reds_r")#, cmap=cmap_type) ## Where ? ,axis=True
plt.axis("on")
plt.margins(5, 0)
#plt.show()
plt.imshow(green_dog_img,cmap="Greens_r")#, cmap=cmap_type)
#plt.show()
plt.imshow(blue_dog_img,cmap="Blues_r")#, cmap=cmap_type)
#plt.show()
plt.imshow(dog_img)#, cmap=cmap_type)
#plt.show()
#RGB to GRAY
from skimage.color import rgb2gray
gray_dog_img = rgb2gray(dog_img)
plt.imshow(gray_dog_img)#
plt.show()
#
#rescale_dog_img = rescale(dog_img,1.0/4.0, anti_aliasing=True) # Notice change in X AXIS Pixel SIZE 
# X Axis - 700 to >> 170 
rescale_dog_img = rescale(dog_img,2.0/3.0, anti_aliasing=True) # Notice change in X AXIS Pixel SIZE 
# X Axis - 700 to >> 145 
plt.imshow(rescale_dog_img)#
plt.show()
#
print("--dog_img.shape---",dog_img.shape)
print("--dog_img.ndim---",dog_img.ndim) # 3D
print("--dog_img.size---",dog_img.size) # The number of pixels - SIZE --dog_img.size--- 2099160
resize_dog_img = rescale(dog_img,(4,4,1), anti_aliasing=True) # 
print("--resize_dog_img.shape---",resize_dog_img.shape)
print("--resize_dog_img.ndim---",resize_dog_img.ndim) # 3D
print("--resize_dog_img.size---",resize_dog_img.size) # The number of pixels - SIZE --dog_img.size--- 2099160
# X AXIS --- 700 >> 2500
# ERRORS Out >> resize_dog_img = rescale(dog_img,(200,200,3), anti_aliasing=True) # 
# MemoryError: Unable to allocate 706. GiB for an array with shape (53800, 146800, 12) and data type float64
plt.imshow(resize_dog_img)#
plt.show()
"""
--dog_img.shape--- (269, 734, 4)
--dog_img.ndim--- 3
--dog_img.size--- 789784
--resize_dog_img.shape--- (1076, 2936, 4)
--resize_dog_img.ndim--- 3
--resize_dog_img.size--- 12636544
"""
#
# downScaled_dog_img = downscale_local_mean(dog_img,(4,3)) #  If Not - 3 >> ValueError: `block_size` must have the same length as `image.shape`.
# plt.imshow(downScaled_dog_img)#
# plt.show()
## SOURCE >> https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html












