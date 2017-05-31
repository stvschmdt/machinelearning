import os
from scipy import ndimage, misc
import re
import numpy as np

def reader():
    d_images = []
    c_images = []
    for root, dirnames, filenames in os.walk("train/"):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                image = ndimage.imread(filepath, mode="RGB")
                image_resized = misc.imresize(image, (28, 28))
                if re.search("dog", filename):
                    d_images.append(image_resized)
                else:
                    c_images.append(image_resized)
                if (len(d_images) + len(c_images))% 1000 == 0:
                    print 'read %s dog and %s cat images'%(len(d_images), len(c_images))

    print 'read %s dog and %s cat images'%(len(d_images), len(c_images))
    return np.array(d_images), np.array(c_images)

if __name__ == '__main__':
    reader()
