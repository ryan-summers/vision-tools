import cv2
import numpy as np
import argparse
import operator
from PIL import Image
import pygame, sys
import time
pygame.init()

def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return ( topleft + bottomright )

def get_crop(image):
    px = pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB");
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower

    print('Lower: {} Upper: {} Left: {} Right: {}'.format(lower, upper, left, right))
    return image[upper:lower, left:right]

def search(model, image, r_step = .2, c_step = .2, start_size = (20, 30), scale = 0.25, display=False):

    ###########################
    # Generate the HOG model. #
    ###########################

    # Window size is equal to test image size because we want 1
    # descriptor per training image.
    winSize = (20, 20)

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

    # Calculate the width and height of the image
    height = np.size(image, 0)
    width = np.size(image, 1)

    detected = image.copy()
    detected_objects = list()

    rows, columns = image.shape[:2]

    # Generate all testable sizes
    sizes = [start_size]
    while sizes[-1][1] + start_size[1] < height and sizes[-1][0]  + start_size[0]< width:
        grow_w = sizes[-1][0] * scale
        grow_h = sizes[-1][1] * scale
        if grow_w < columns / 75.0:
            grow_w = columns / 75.0
        if grow_h < rows / 75.0:
            grow_h = rows / 75.0
        if grow_w > .2 * columns:
            grow_w = .2 * columns
        if grow_h > .2 * rows:
            grow_h = .2 * rows
        size = (sizes[-1][0] + grow_w, sizes[-1][1] + grow_h)
        sizes.append(size)

    detections = 0

    # Subsample the image with the testing sizes and rescale to 25x100
    for size in sizes:

        # Slide the test window across the row
        col_step = c_step * size[0]
        if col_step > columns / 20.0:
            col_step = columns / 20.0
        if col_step < columns / 25:
            col_step = columns / 25
        for col in np.arange(0, width - size[0], col_step):

            # slide the test window across the column for each row position
            row_step = c_step * size[1]
            if row_step > rows / 5.0:
                row_step = rows / 5.0
            if row_step < rows / 25:
                row_step = rows / 25
            for row in np.arange(0, height - size[1], row_step):

                # Test the cropped image on the model.
#                cropped = get_crop(image)
#                cropped = cv2.resize(cropped, winSize)

                cropped = cv2.resize(image[int(row):int(row + size[1]), int(col):int(col + size[0])], winSize)
                if display is True:
                    copy = detected.copy()
                    cv2.rectangle(copy, (int(col), int(row)), (int(col + size[0]), int(row + size[1])), (0, 0, 255), 2)
                    cv2.imshow('Searching', copy)
                    cv2.imshow('Cropped', cropped)
                    cv2.waitKey(1)
                hog_features = hog.compute(cropped)
                hog_features = np.transpose(hog_features)
                response = model.predict(hog_features)
                res = np.squeeze(response[1])

                if res == 1:
                    detected_objects.append([int(col), int(col + size[0]), int(row), int(row + size[1])])

    # Apply local maxima suppression to remove overlapping detections.
    final_objects = list(list())
    for obj in detected_objects:
        if len(final_objects) == 0:
            final_objects.append(obj)
        else:
            obj_center = ((obj[0] + obj[1]) / 2, (obj[2] + obj[3])/2)
            added = False
            for i in range(0, len(final_objects)):
                # Because the objects are detected in ascending size, always overwrite the smaller sized object.
                if obj_center[0] in range(final_objects[i][0], final_objects[i][1]) and obj_center[1] in range(final_objects[i][2], final_objects[i][3]):
                    final_objects[i]= obj
                    added = True
                    break
            if added is False:
                final_objects.append(obj)

    for i in range(0, len(final_objects)):
        cv2.rectangle(detected, (final_objects[i][0], final_objects[i][2]), (final_objects[i][1], final_objects[i][3]), (0, 255, 0), 2)

    return len(final_objects), detected

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search an image given a specific model for an object.')
    parser.add_argument('model', help='The SVM data model file')
    parser.add_argument('image', help='The image to search')
    parser.add_argument('--display', help='Graphically show where searches occur.', action='store_true')

    args = parser.parse_args()

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow('Searching', image)
    cv2.waitKey(500)

    model = cv2.ml.SVM_load(args.model)

    start = time.time()
    detections, final_image = search(model, image, display=args.display)
    print('Detected {} objects in {} seconds.'.format(detections, time.time() - start))
    cv2.imshow('Final', final_image)
    cv2.waitKey()
