import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE_WIDTH=28
IMAGE_HEIGHT=28

def transform_img_to_nparray(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

if __name__ == '__main__':
    print 'showing test harness for 2 jpg of different sizes -> normalized'
    #testing harness
    p1 = 'train/cat.0.jpg'
    p2 = 'train/cat.1.jpg'
    #before
    img1 = cv2.imread(p1, cv2.IMREAD_COLOR)
    #plt.imshow(img1)
    #plt.show()
    #after
    img1 = transform_img_to_nparray(img1, img_width=250, img_height=250)
    #plt.imshow(img1)
    #plt.show()
    #before
    img2 = cv2.imread(p2, cv2.IMREAD_COLOR)
    #plt.imshow(img2)
    #plt.show()
    #after
    img2 = transform_img_to_nparray(img2, img_width=250, img_height=250)
    #plt.imshow(img2)
    #plt.show()
