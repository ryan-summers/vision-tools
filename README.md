=== ML Vision Tools ===

Description of files and their purposes:

- BuoyDetector.py
    Work in progress python class

- cropper,py
    Will take an input image, display it, allow you to select
    crop regions by clicking and dragging and releasing, and then
    save the output cropped image. This is useful for generating
    machine learning data sets.

- hog_buoy.py
    Used for training an SVM model. The model is then saved for
    reuse and testing later.

- search_buoy.py
    Attempts to search for an image for an object. A bounding box
    will be drawn in green.

- subsample.py
    Generates a number of randomly sized images from a given
    input image. This is useful for making large negative data
    sets given a set of images known to not contain the desired
    object.
