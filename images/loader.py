import os
from scipy import ndimage, misc
import re

images = []
for root, dirnames, filenames in os.walk("train/"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (28, 28))
            images.append(image_resized)
	    if len(images) % 1000 == 0:
		print 'read %s images'%len(images)

print 'read %s images'%len(images)
