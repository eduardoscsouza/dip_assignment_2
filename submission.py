'''
Name: Eduardo Santos Carlos de Souza
USP Number: 9293481
Course: SCC0251/SCC5830 - Image Processing - 1st Semester of 2020

Assignment 2: Image Enhancement and Filtering
'''



#Lib imports
import numpy as np
from imageio import imread, imwrite
from itertools import product



def conv_2d(img, fil, pad=True, pad_mode='constant', **pad_kwargs):
    orig_shape = img.shape
    pad_shape = (((fil.shape[0]-1)//2, fil.shape[0]//2), ((fil.shape[1]-1)//2, fil.shape[1]//2)) if pad else ((0, 0), (0, 0))
    img = np.pad(img, pad_shape, mode=pad_mode, **pad_kwargs)

    slices_centers = [(i, j) for i, j in product(range(pad_shape[0][0], pad_shape[0][0]+orig_shape[0]), range(pad_shape[1][0], pad_shape[1][0]+orig_shape[1]))]
    slices = np.stack([img[i-pad_shape[0][0]:i+pad_shape[0][1]+1, j-pad_shape[1][0]:j+pad_shape[1][1]+1].flatten() for i, j in slices_centers])
    img = np.matmul(slices, fil.flatten()).reshape(orig_shape)
    return img



def bilateral(img, fil_size, sig_s, sig_rd):
    start, finish = -((fil_size-1)//2), (fil_size//2)
    points = np.stack([(i, j) for i, j in product(range(start, finish+1), range(start, finish+1))]).reshape((fil_size, fil_size, 2))
    dists = np.linalg.norm(points, axis=2)
    gauss = np.exp(-(dists**2) / (2*sig_s**2)) /  (2*np.pi*sig_s**2)
    
    print(dists)

def unsharp(img, c, ker):
    pass

def vignette(img, sig_row, sig_col):
    pass



#Definition of the Root Squared Error Function
def rse(anchor, compare):
    return np.sqrt(np.sum(np.square(anchor-compare)))



#Main Function
if __name__ ==  '__main__':
    bilateral(None, 5, 3, 3)
    '''
    #Reading input parameters
    img_filename = str(input()).rstrip()
    method = int(input())
    save = bool(int(input()))
    if method == 1:
        fil_size, sig_s, sig_rd = int(input()), float(input()), float(input())
    if method == 2:
        c, ker = float(input()), int(input())
    if method == 3:
        sig_row, sig_col = float(input()), float(input())

    #Read the image and convert to float for the mathematical operations
    img = imread(img_filename).astype(np.float64)
    #Apply the specified method
    if method == 1:
        res_img = bilateral(img, fil_size, sig_s, sig_rd)
    elif method == 2:
        res_img = unsharp(img, c, ker)
    else:
        res_img = vignette(img, sig_row, sig_col)

    #Save if necessary
    if save:
        imwrite("output_img.png", res_img.astype(np.uint8))

    #Print the RSE
    print("%.4f" % rse(res_img, img))
    '''