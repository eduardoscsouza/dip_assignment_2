'''
Name: Eduardo Santos Carlos de Souza
USP Number: 9293481
Course: SCC0251/SCC5830 - Image Processing - 1st Semester of 2020

Assignment X: AAA
'''



#Lib imports
import numpy as np
from imageio import imread, imwrite



#Definition of the Root Squared Error Function
def rse(anchor, compare):
    return np.sqrt(np.sum(np.square(anchor-compare)))



#Main Function
if __name__ ==  '__main__':
    #Reading input parameters
    img_filename = str(input()).rstrip()
    save = bool(int(input()))

    #Read the image and convert to float for the mathematical operations
    img = imread(img_filename).astype(np.float64)

    #Save if necessary
    if save:
        imwrite("output_img.png", res_img.astype(np.uint8))

    #Print the RSE
    print("%.4f" % rse(res_img, img))