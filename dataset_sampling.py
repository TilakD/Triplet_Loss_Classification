"""reduce the size of training set by random sampling"""

import argparse
import glob
import os

import numpy as np
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/fruits360/train',
                    help="Source dataset directory")
parser.add_argument('--destination', default='data/fruits360_reduced/train',
                    help="Destination directory")
parser.add_argument('--size', default='10',
                    help="Number of observations per class in destination folder")
parser.add_argument('--mode', default='copy',
                    help="File transfer mode (copy or move)")


if __name__ == '__main__':

    args = parser.parse_args()

    for d in os.listdir(args.dataset_dir):
        class_dir = os.path.join(args.dataset_dir, d)
        print("\n * class dir {}".format(class_dir))
        if os.path.isdir(class_dir):
            # get all images from each class folder
            image_list = glob.glob(class_dir+"/*")
            # select random images to keep
            size = int(args.size)
            if size < len(image_list):
                keep = np.random.choice(image_list, size, replace=False)
            else:
                keep = image_list

            # create destination directory
            if not os.path.exists(args.destination):
                os.makedirs(args.destination)
            
            # create destination class directory
            os.makedirs(os.path.join(args.destination, os.path.basename(class_dir)))

            # copy or move the images selected in the destination folder
            for i in keep:
                if args.mode == "copy":
                    print("Copy:", i,"-->", os.path.join(args.destination, os.path.basename(class_dir), os.path.basename(i)))
                    shutil.copy2(i, os.path.join(args.destination, os.path.basename(class_dir)))
                if args.mode == "move":
                    print("Move:", i,"-->", os.path.join(args.destination, os.path.basename(class_dir), os.path.basename(i)))
                    shutil.move(i, os.path.join(args.destination, os.path.basename(class_dir)))


