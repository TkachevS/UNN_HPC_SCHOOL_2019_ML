"""

Inference engine detector sample
 
"""
import sys
import cv2
import argparse
import logging as log
sys.path.append('../src')
from ie_detector import InferenceEngineDetector
import time
def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pX', '--pathXML', help = 'path to input XML', type = str)
    parser.add_argument('-pB', '--pathBin', help = 'path to input bin', type = str)
    parser.add_argument('-t', '--type', help = 'type of device', type = str)
    parser.add_argument('-l', '--pathLib', help = 'path to library', type = str)  
    parser.add_argument('-i', '--image', help = 'path to image', type = str)
    parser.add_argument('-v',"--video", help="path to video file", type = str)
    return parser

def main():
    log.basicConfig(format = "[%(levelname)s] %(message)s", level = log.INFO, 
                               stream = sys.stdout)
    log.info("Hello object detection!")
    args = build_argparse().parse_args()
    ied = InferenceEngineDetector(weightsPath = args.pathBin, configPath = args.pathXML,
                 device = args.type, extension = args.pathLib)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_resized = cv2.resize(frame,(300,300))
                final_image = ied.detect(frame_resized)
                cv2.imshow("Post-processing image", final_image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
    elif args.image:
        cap = cv2.imread(args.image)        
        final_image = ied.detect(cap)
        cv2.imshow("Post-processing image", final_image)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    return 

if __name__ == '__main__':
    sys.exit(main()) 