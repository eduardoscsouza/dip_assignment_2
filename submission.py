'''
Name: Eduardo Santos Carlos de Souza
USP Number: 9293481
Course: SCC0251/SCC5830 - Image Processing - 1st Semester of 2020

Assignment 2: Image Enhancement and Filtering
Git Repository: https://github.com/eduardoscsouza/dip_assignment_2
'''



#Lib imports
import numpy as np
from imageio import imread, imwrite



#Function to extract the neighborhood with the shape of the filter for each pixel
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
    #Stack the list of slices into a ndarray
    slices = np.stack(slices)

    return slices, res_shape

#Function to execute a 2D convolution
def conv_2d(img, fil, pad=True, pad_mode='constant', **pad_kwargs):
    #Get the neighborhoods of each pixel, and the resulting image shape
    slices, res_shape = __get_neighs__(img, fil.shape, pad, pad_mode, **pad_kwargs)
    #Changes in shape for easier numpy calculations
    fil = fil.flatten()[:, np.newaxis]
    #Calculate the convolution through a matrix multiplication, and reshape to the resulting image size
    img = np.matmul(slices, fil).reshape(res_shape)

    return img



#Function to calculate the gaussian kernels
def __get_gauss__(vals, sig):
    return np.exp(-(vals**2) / (2*sig**2)) /  (2*np.pi*sig**2)

#Function to execute the bilateral filter
def bilateral(img, fil_size, sig_s, sig_r):
    #Calculate the spatial gaussian
    #Get an array with the coordinates of each point for the gaussian function
    center_pix_idx = (fil_size-1)//2
    points_start = np.asarray((-center_pix_idx, -center_pix_idx))[:, np.newaxis, np.newaxis]
    points = np.indices((fil_size, fil_size)) + points_start
    #Get the euclidian distance, apply the gaussian function, and flatten for easier calculations
    spat_gauss = __get_gauss__(np.linalg.norm(points, axis=0), sig_s).flatten()

    #Calculate the gaussian kernel
    #Update center_pix_idx for the flattened array
    center_pix_idx = (center_pix_idx * fil_size) + center_pix_idx
    #Get the neighborhoods of each pixel, and the resulting image shape
    neighs, res_shape = __get_neighs__(img, (fil_size, fil_size))
    #Calculate the gaussian function on each neighborhood, with the center subtracted
    kern_gauss = __get_gauss__(neighs - neighs[:, center_pix_idx][:, np.newaxis], sig_r)

    #Calculate the bilateral filter
    weights = kern_gauss * spat_gauss
    img = np.sum(weights * neighs, axis=1) / np.sum(weights, axis=1)

    return img.reshape(res_shape)



#Function to set the range of the pixels between 0 and 255
def __normalize__(img):
    img_min, img_max = np.min(img), np.max(img)
    img = (img - img_min) * (255.0 / (img_max - img_min))
    return img

#Function to execute the unsharp filter
def unsharp(img, c, ker):
    #Get the chosen filter
    if ker == 1:
        fil = np.ones((3, 3)) * -1
        fil[0, 0] = fil[0, 2] = fil[2, 0] = fil[2, 2] = 0
        fil[1, 1] = 4
    else:
        fil = np.ones((3, 3)) * -1
        fil[1, 1] = 8

    #Apply the convolution and normalize
    res_img = __normalize__(conv_2d(img, fil))
    #Add the original image and normalize
    res_img = __normalize__(c*(res_img) + img)

    return res_img



#Function to execute the vignette filter
def vignette(img, sig_row, sig_col):
    #Get the gaussian kernel for the rows, and change shape for future calculations
    row_gauss = __get_gauss__(np.arange(-((img.shape[0]-1)//2), (img.shape[0]//2)+1), sig_row)[:, np.newaxis]
    #Get the gaussian kernel for the columns, and change shape for future calculations
    col_gauss = __get_gauss__(np.arange(-((img.shape[1]-1)//2), (img.shape[1]//2)+1), sig_col)[np.newaxis, :]
    #Get the weight matrix through matrix multiplication
    weights = np.matmul(row_gauss, col_gauss)
    #Calculate the point-wise multiplication and normalize
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