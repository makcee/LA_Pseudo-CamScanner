from utils import *
import numpy as np


def warpPerspective(img, transform_matrix, output_width, output_height):
   
    res = np.empty([output_width,output_height,3])
    vec3by1 = np.empty([3,1])
    for i in range (0,img.shape[0]):
        for j in range (0,img.shape[1]):
            vec3by1 = np.matmul(transform_matrix, np.array([i,j,1]))
            xSec = vec3by1[0]/vec3by1[2]
            ySec = vec3by1[1]/vec3by1[2]
            xSec = int(xSec)
            ySec = int(ySec)
            if (xSec >= 0 and xSec < 300 and ySec >= 0 and ySec < 400):
                res[xSec][ySec] = img[i][j]
    
    return res


def grayScaledFilter(img):
    return Filter(img, np.array([[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33]]))


def crazyFilter(img):
    return Filter(img, np.array([[0,1,1],[1,0,0],[0,0,0]]))


def customFilter(img):
    filterMatrix = np.array([[0,0,1],[0,1,0],[1,0,0]])
    customImage = Filter(img, filterMatrix)
    showImage(customImage, title="User Filter") 
    InvertedCustomImage = Filter(customImage, np.linalg.inv(filterMatrix))
    showImage(InvertedCustomImage, title="Inverted User Filter") 


def scaleImg(img, scale_width, scale_height):
    res = np.empty([scale_width * img.shape[0], scale_height * img.shape[1], 3])
    for i in range (0,scale_width * img.shape[0]):
        for j in range (0,scale_height * img.shape[1]):
            newx = i * (1 / scale_width)
            newx = int(newx)
            newy = j * (1 / scale_height)
            newy = int(newy)
            res[i][j] = img[newx][newy]
    
    return res


def cropImg(img, start_row, end_row, start_column, end_column):
    res = np.empty([end_column - start_column, end_row - start_row, 3])
    for i in range (start_column, end_column):
        for j in range (start_row, end_row):
            res[i - start_column][j - start_row] = img[i][j]
    
    return res


if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    # You can change width and height if you want
    width, height = 300, 400

    showImage(image_matrix, title="Input Image")

    # TODO : Find coordinates of four corners of your inner Image ( X,Y format)
    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[105, 214], [377, 179], [157, 645], [493, 570]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective(warpedImage)

    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")

    customFilter(warpedImage)

    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 2, 3)
    showImage(scaledImage, title="Scaled Image")
