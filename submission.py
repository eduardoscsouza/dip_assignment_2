'''
Name: Eduardo Santos Carlos de Souza
USP Number: 9293481
Course: SCC0251/SCC5830 - Image Processing - 1st Semester of 2020

Assignment 2: Image Enhancement and Filtering
'''



#Lib imports
import numpy as np
from imageio import imread, imwrite



def __get_neighs__(img, fil_shape, pad=True, pad_mode='constant', **pad_kwargs):
    #Calculate how many pixels the convolution considers in regards to the central pixel in all 4 directions
    conv_borders = (((fil_shape[0]-1)//2, fil_shape[0]//2), ((fil_shape[1]-1)//2, fil_shape[1]//2))
    #Calculate how many pixels are not applicable at the edges of the image
    conv_borders_loss = (conv_borders[0][0]+conv_borders[0][1], conv_borders[1][0]+conv_borders[1][1])

    if pad:
        #Stores the unaltered resulting shape for the final image
        res_shape = img.shape
        #Pads the image with exactly the necessary number of pixels
        img = np.pad(img, conv_borders, mode=pad_mode, **pad_kwargs)
    else:
        #Stores the reduced resulting shape for the final image
        res_shape = (img.shape[0]-conv_borders_loss[0], img.shape[1]-conv_borders_loss[1])

    #Calculate on how many pixels the convolution will be applied, in each dimension
    conv_area_size = (img.shape[0]-conv_borders_loss[0], img.shape[1]-conv_borders_loss[1])
    #Calculate the indexes of the first pixels on wich the convolution will be applied, in each dimension,
    #and change to a ndarray of specific shape for easier calculations
    conv_area_begin = np.asarray((conv_borders[0][0], conv_borders[1][0]))[:, np.newaxis, np.newaxis]
    #Calculate the indexes of the pixels on which the convolution will be applied
    slices_centers = np.indices(conv_area_size) + conv_area_begin
    #Change in shape so the variable is effectively a list of pairs of indexes
    slices_centers = slices_centers.reshape(slices_centers.shape[0], slices_centers.shape[1] * slices_centers.shape[2]).transpose()

    #For each central pixel of the convolution operation, take a slice of the image with all the convolution area for that pixel
    slices = [img[i-conv_borders[0][0]:i+conv_borders[0][1]+1, j-conv_borders[1][0]:j+conv_borders[1][1]+1].flatten() for i, j in slices_centers]
    slices = np.stack(slices)

    return slices, res_shape

def conv_2d(img, fil, pad=True, pad_mode='constant', **pad_kwargs):
    #Get the neighborhoods of each pixel, and the resulting image shape
    slices, res_shape = __get_neighs__(img, fil.shape, pad, pad_mode, **pad_kwargs)
    #Changes in shape for easier numpy calculations
    fil = fil.flatten()
    #Calculate the convolution through a matrix multiplication, and reshape to the resulting image size
    img = np.matmul(slices, fil).reshape(res_shape)

    return img



def __get_gauss__(vals, sig):
    return np.exp(-(vals**2) / (2*sig**2)) /  (2*np.pi*sig**2)

def bilateral(img, fil_size, sig_s, sig_r):
    center_pix_idx = (fil_size-1)//2
    points_start = (-center_pix_idx, -center_pix_idx)
    points_start = np.asarray(points_start)[:, np.newaxis, np.newaxis]
    points = np.indices((fil_size, fil_size)) + points_start
    spat_gauss = __get_gauss__(np.linalg.norm(points, axis=0), sig_s).flatten()

    center_pix_idx = (center_pix_idx * fil_size) + center_pix_idx
    neighs, res_shape = __get_neighs__(img, (fil_size, fil_size))
    kern_gauss = __get_gauss__(neighs - neighs[:, center_pix_idx][:, np.newaxis], sig_r)

    weights = kern_gauss * spat_gauss
    img = np.sum(weights * neighs, axis=1) / np.sum(weights, axis=1)

    return img.reshape(res_shape)



def __normalize__(img):
    img = (img - np.min(img)) * (255.0 / np.max(img))
    return img

def unsharp(img, c, ker):
    if ker == 1:
        fil = np.ones((3, 3)) * -1
        fil[0, 0] = fil[0, 2] = fil[2, 0] = fil[2, 2] = 0
        fil[1, 1] = 4
    else:
        fil = np.ones((3, 3)) * -1
        fil[1, 1] = 8

    res_img = __normalize__(conv_2d(img, fil))
    res_img = __normalize__(c*(res_img) + img)

    return res_img



def vignette(img, sig_row, sig_col):
    row_gauss = __get_gauss__(np.arange(-((img.shape[0]-1)//2), (img.shape[0]//2)+1), sig_row)[:, np.newaxis]
    col_gauss = __get_gauss__(np.arange(-((img.shape[1]-1)//2), (img.shape[1]//2)+1), sig_col)[np.newaxis, :]
    weights = np.matmul(row_gauss, col_gauss)
    img = __normalize__(img * weights)

    return img



#Definition of the Root Squared Error Function
def rse(anchor, compare):
    return np.sqrt(np.sum(np.square(anchor-compare)))



#Main Function
if __name__ ==  '__main__':
    #Reading input parameters
    img_filename = str(input()).rstrip()
    method = int(input())
    save = bool(int(input()))

    #Read the image and convert to float for the mathematical operations
    img = imread(img_filename).astype(np.float64)
    #Read extra inputs and apply the specified method
    if method == 1:
        fil_size, sig_s, sig_r = int(input()), float(input()), float(input())
        res_img = bilateral(img, fil_size, sig_s, sig_r)
    elif method == 2:
        c, ker = float(input()), int(input())
        res_img = unsharp(img, c, ker)
    else:
        sig_row, sig_col = float(input()), float(input())
        res_img = vignette(img, sig_row, sig_col)

    #Save if necessary
    if save:
        imwrite("output_img.png", res_img.astype(np.uint8))

    #Print the RSE
    print("%.4f" % rse(res_img, img))