import argparse
import operator
import cv2
import numpy
import os
import glob

def train(model, dirname, save_file='svm_model.dat'):
    images = list()
    labels = list()

    winSize = (20, 20)

    for f in glob.glob(dirname + '/positives/*.jpg'):
        # Resize all test images to 25x25
        image = cv2.imread(f)
        image = cv2.resize(image, winSize)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        labels.append(1)

    print('Done labelling positives')

    for f in glob.glob(dirname + '/negatives/*.jpg'):
        # Resize all test images to 25x25
        image = cv2.imread(f)
        image = cv2.resize(image, winSize)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        labels.append(-1)

    print('Done labelling negatives')

    ###########################
    # Generate the HOG model. #
    ###########################

    # Window size is equal to test image size because we want 1
    # descriptor per training image.

    # The cell size of each feature should be decently large
    cellSize = (5, 5)
    # Block size is typically 2 cell sizes.
    blockSize = tuple(map(operator.add, cellSize, cellSize))
    # Block stride is typically half of block size to have 50% block overlap.
    blockStride = cellSize

    # Produce 9 bins for bin size of 20 degrees from 0 to 180.
    nbins = 9

    # Specifies that gradients should be signed.
    useSignedGradients = True

    # These parameters rarely need adjusting and are taken from
    # LearnOpenCV's website.
    derivAperature = 1
    winSigma = -.1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nLevels = 64;

    # Generate the HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
            cellSize, nbins, derivAperature, winSigma,
            histogramNormType, L2HysThreshold, gammaCorrection,
            nLevels, useSignedGradients);

    print('Done creating HOG descriptor')

    # Generate HOG descriptors for the images.
    hog_descriptors = list();
    for image in images:
        hog_descriptors.append(hog.compute(image))

    # Squeeze dimensions down with numpy
    hog_descriptors = numpy.squeeze(hog_descriptors)
    print('Hog descriptors: {}'.format(hog_descriptors.shape))

    print('Training model.');
    model.train(hog_descriptors, cv2.ml.ROW_SAMPLE, numpy.array(labels));

    print('Saving trained model to {}'.format(save_file))
    model.save(save_file)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SVM model.');
    parser.add_argument('path', help='Path to the test image file '
            'locations. Should contain positive/ and negative/ subdirectories')

    args = parser.parse_args();

    # Generate an SVM model.
    model = cv2.ml.SVM_create()
    model.setGamma(0.50625)
    model.setC(12.5)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)

    # Train the SVM model.
    train(model, args.path)

    print('Finished training.')
