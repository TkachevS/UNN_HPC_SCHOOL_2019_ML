import cv2
import math
import logging as log
class ImageFilter():
    def __init__(self, gray = False, shape = None, crop = False):
        self.gray = gray
        self.crop = crop
        self.shape = shape
    
    def process_image(self, image):
        if self.shape:
            image = cv2.resize(image, dsize = self.shape)
        if self.gray:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if self.crop:
            half_width = image.shape[0]//2
            half_length = image.shape[1]//2
            minimum = min(half_width, half_length)
            center = [half_width, half_length]
            image = image[center[0]-minimum:center[0]+minimum,
                            center[1]-minimum:center[1]+minimum]
            log.info(str(center))
            log.info(str(center[1]) + "+-"+str(minimum))
            log.info(str(center[0]) + "+-"+str(minimum))    
        return image
    
    