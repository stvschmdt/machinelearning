import os
from scipy import ndimage, misc
import re
import numpy as np
from logger import Logging

class Reader():
    def __init__(self):
        self.logger = Logging()

    def read_images(self, fname, sep):
        d_images = []
        c_images = []
        for root, dirnames, filenames in os.walk(fname):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename)
                    image = ndimage.imread(filepath, mode="RGB")
                    image_resized = misc.imresize(image, (28, 28))
                    if re.search(sep, filename):
                        d_images.append(image_resized)
                    else:
                        c_images.append(image_resized)
                    if (len(d_images) + len(c_images))% 1000 == 0:
                        self.logger.info('read %s/%s %s/not %s images'%(len(d_images), len(c_images), sep, sep))

        self.logger.info('read %s dog and %s cat images'%(len(d_images), len(c_images)))
        return np.array(d_images), np.array(c_images)

if __name__ == '__main__':
    r = Reader()
    arr = r.read_images('../../store/images_data/train', 'dog')
