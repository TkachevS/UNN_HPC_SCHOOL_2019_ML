import sys
import cv2
import logging as log
import argparse

sys.path.append('../src')
from imagefilter import ImageFilter

def build_argparse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help = 'your input \
            parameter (image, video, etc)', type = str)
    parser.add_argument('-w', '--width', help = 'your width parameter for image', type = int)
    parser.add_argument('-l', '--length', help = 'your length parameter for image', type = int)
    parser.add_argument('-c', '--crop', help = 'your crop-filter using for image', type = bool)
    parser.add_argument('-g', '--gray', help = 'your crop-filter using for image', type = bool)
    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", 
                             level=log.INFO, stream=sys.stdout)
    log.info("Hello image filtering")
    args = build_argparse().parse_args()
    imagePath = args.input
    log.info(imagePath)
    image_source = cv2.imread(imagePath)
    log.info(image_source.shape)
    imageFilter = ImageFilter(shape = (args.width, args.length), crop = args.crop, gray = args.gray)
    final_image = imageFilter.process_image(image_source)
    cv2.imshow("Post-processing image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    sys.exit(main()) 