'''
Name: Eduardo Santos Carlos de Souza
USP Number: 9293481
Course: SCC0251/SCC5830 - Image Processing - 1st Semester of 2020

Assignment 2: Image Enhancement and Filtering
'''



#Lib imports
import numpy as np
from imageio import imread, imwrite



def bilateral(img, fil_size, sig_s, sig_rd):
    pass

def unsharp(img, c, ker):
    pass

def vignette(img, sig_row, sig_col):
    pass



#Definition of the Root Squared Error Function
def rse(anchor, compare):
    return np.sqrt(np.sum(np.square(anchor-compare)))



#Main Function
if __name__ ==  '__main__':
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