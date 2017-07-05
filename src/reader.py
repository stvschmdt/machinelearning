import os
from scipy import ndimage, misc
import re
import numpy as np
from logger import Logging

class Reader():
    def __init__(self):
        self.logger = Logging()

    def average(self, pixel):
        return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

    def read_images(self, fname, sep, grayscale=None, test=None):
        self.d_images = []
        self.c_images = []
        self.file_labels = []
        for root, dirnames, filenames in os.walk(fname):
            for filename in sorted(filenames, key=lambda x: int(x.split('.')[0])):
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename)
                    self.file_labels.append(filename.split('.')[0])
                    image = ndimage.imread(filepath, mode="RGB")
                    image_resized = misc.imresize(image, (28, 28))
                    #check if this is from one class (dog in test)
                    if re.search(sep, filename):
                        #are we making this grayscale (x,784) or not
                        if grayscale:
				grey = np.zeros((image_resized.shape[0], image_resized.shape[1])) # init 2D numpy array
				# get row number
				for rownum in range(len(image_resized)):
   				    for colnum in range(len(image_resized[rownum])):
      				        grey[rownum][colnum] = self.average(image_resized[rownum][colnum])
                                self.d_images.append(grey.reshape(784))
                        #not
                        else:
                            self.d_images.append(image_resized)
                    #cat in testing suite
                    else:
                        if grayscale:
				grey = np.zeros((image_resized.shape[0], image_resized.shape[1])) # init 2D numpy array
				# get row number
				for rownum in range(len(image_resized)):
   					for colnum in range(len(image_resized[rownum])):
      						grey[rownum][colnum] = self.average(image_resized[rownum][colnum])

                                self.c_images.append(grey.reshape(784))
                        else:
                            self.c_images.append(image_resized)
                    if (len(self.d_images) + len(self.c_images))% 1000 == 0:
                        self.logger.info('read %s/%s %s/not %s images'%(len(self.d_images), len(self.c_images), sep, sep))
                        if test:
                            return

        self.logger.info('read %s dog and %s cat images'%(len(self.d_images), len(self.c_images)))
        self.d_images = np.array(self.d_images)
        self.c_images = np.array(self.c_images)
        return self.d_images, self.c_images

if __name__ == '__main__':
    r = Reader()
    r.read_images('../../store/images_data/train', 'dog', True,True)
    print type(r.d_images), type(r.c_images)
    x = np.array(r.d_images)
    y = np.array(r.c_images)
    print type(x), x.shape
    #print x[0].shape, x[0][0].shape, x[0][0][0].shape

