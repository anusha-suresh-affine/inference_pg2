# imports required
import cv2
import os
import itertools
import numpy as np

import matplotlib.image as mpimg
from skimage import transform
import logging

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# adding logging functionality

# setting up the logging configuration here.
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s :: %(levelname)s :: %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')
                    # filename='logs.txt'

logger = logging.getLogger('apps')
logger.info(str(os.getcwd()))


def classification(image, model, input_folder):
    """
    Classify images into defect/no-defect

    :param report_dict: dict to append classification details
    :param image: list of images for running classification
    :param model: classification model

    :return: updated report_dict with classification result details

    """
    logger.info("Image classification started")
    logger.info("Loading Image classification model")

    report_dict = {}
    ij = mpimg.imread(os.path.join(input_folder, image))
    np_image = np.array(ij).astype('float32')/255
    np_image = transform.resize(np_image, (512, 512, 3))
    np_image = np.expand_dims(np_image, axis=0)
    try:
        resp = model.predict(np_image)
    except Exception:
        logger.info("Image is not processed for classification model: " + image, exc_info=True, stack_info=True)
    report_dict[image] ={'is_defective': True if resp[0][0] <= 0.5 else False,\
                                                        'confidence': round(resp[0][0], 4)}

    return report_dict
    