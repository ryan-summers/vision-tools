import cv2
import numpy
import glob

class BuoyDetector:
    def __init__(self, name, start_size=(20, 30), r_step = .2, c_step = .2, scale = 0.25):
        self.name = name
        self.model = cv2.ml.SVM_create();
        self.model.setGamma(0.50625)
        self.model.setC(12.5)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

        # For buoys, a window size of 20x20 and cell sizes of 5x5
        # work well for detection.
        self.winSize = (20, 20)

        # The cell size of each feature should be decently large. Empirically,
        # a value of 5x5 has worked well for buoy detection. This
        # results in 16 cells within an image.
        cellSize = (5, 5)

        # Block size is typically 2 cell sizes.
        blockSize = tuple(map(operator.add, cellSize, cellSize))

        # Block stride is typically half of block size to have
        # 50% block overlap in normaliation steps for HOG.
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

        # Generate the HOG analyzer.
        self.hog = cv2.HOGDescriptor(self.winSize, blockSize, blockStride,
                cellSize, nbins, derivAperature, winSigma,
                histogramNormType, L2HysThreshold, gammaCorrection,
                nLevels, useSignedGradients);

        # Generate all testable sizes for image searching.
        self.sizes = [start_size]
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

    def load(self, svm_data):
        self.model = cv2.ml.SVM_load(svm_data)

    def train(self, positives_dir, negatives_dir):
        images = list()
        labels = list()

        for f in glob.glob('{}/*.jpg'.format(positives_dir)):
            # Resize all test images to the window size
            image = cv2.imread(f)
            image = cv2.resize(image, self.winSize)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            labels.append(1)

        for f in glob.glob('{}/*.jpg'.format(negatives_dir)):
            # Resize all test images to the window size
            image = cv2.imread(f)
            image = cv2.resize(image, self.winSize)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            labels.append(-1)


        # Generate HOG descriptors for each image in the training set.
        hog_descriptors = list();
        for image in images:
            hog_descriptors.append(self.hog.compute(image))

        # Squeeze descriptor dimensions down with numpy
        hog_descriptors = numpy.squeeze(hog_descriptors)

        # Train the model with the provided data set.
        self.model.train(hog_descriptors, cv2.ml.ROW_SAMPLE,
                numpy.array(labels));

    def search(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

        # Subsample the image with the testing sizes and rescale to 25x100
        for size in self.sizes:

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

                    cropped = cv2.resize(image[int(row):int(row + size[1]), int(col):int(col + size[0])], winSize)

                    hog_features = self.hog.compute(cropped)
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
