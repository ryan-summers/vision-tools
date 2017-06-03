import argparse
import cv2
import glob
import operator
import numpy as np
import os.path
import time
from search_buoy import search

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search an image given a specific model for an object.')
    parser.add_argument('model', help='The SVM data model file', type=str)
    parser.add_argument('image_dir', help='The directory containing images to search', type=str)
    parser.add_argument('--out', help='The directory to save final results to', type=str)
    args = parser.parse_args()

    model = cv2.ml.SVM_load(args.model)

    total_time = 0
    total_detections = 0
    total_tests = 0
    for f in glob.glob('{}/*.jpg'.format(args.image_dir)):
        print('Testing {}'.format(f))
        total_tests = total_tests + 1
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
        t_start = time.time()
        detections, final_image = search(model, image)
        total_time = total_time + (time.time() - t_start)
        if detections != 0:
            total_detections = total_detections + 1
            # print('Detected {} item(s) in {}'.format(detections, f))
            if args.out is not None:
                save_name = '{}/{}-det.jpg'.format(args.out, os.path.splitext(os.path.basename(f))[0])
                cv2.imwrite(save_name, final_image)
                # print('Saved result to {}'.format(save_name))
        else:
            print('Failed to detect items in {}'.format(f))
            if args.out is not None:
                save_name = '{}/{}'.format(args.out, os.path.basename(f))
                cv2.imwrite(save_name, final_image)

    print('Detected objects in {}/{} images. Average of {} seconds per test.'.format(total_detections, total_tests, (total_time) / total_tests))
