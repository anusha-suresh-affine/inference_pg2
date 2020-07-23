# imports required
import time
import cv2
import os
import itertools
import numpy as np

import matplotlib.image as mpimg
from skimage import transform
import logging


from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from collections import defaultdict
from keras import backend as K

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

logger = logging.getLogger('apps')
logger.info(str(os.getcwd()))

# weights for defect index
weights = {'mistracking': 0.5, 'trim': 0.5, 'contamination': 100, 'tear': 20.56, 'wrinkle': 0.5}

# label map for model classes
labels_to_names = {0: 'mistracking', 1: 'trim', 2: 'contamination', 3: 'tear', 4: 'wrinkle'}


def compute_bbox_area(bbox):
    """
    Function to compute area of individual bounding boxes

    :param bbox: bbox coordinates for a predicted defect
    :return: area of the box
    """
    xmin, ymin, xmax, ymax = tuple(bbox)
    boxarea = (xmax - xmin) * (ymax - ymin)
    return float(boxarea)


def compute_intersection(bbox1, bbox2):
    """
    Function to compute the intersection of 2 bounding boxes

    :param bbox1: Box coordinates (xmin, ymin, xmax, ymax)
    :param bbox2: Box coordinates (xmin, ymin, xmax, ymax)
    :return: intersection area between 2 input boxes
    """
    g_xmin, g_ymin, g_xmax, g_ymax = tuple(bbox1)
    d_xmin, d_ymin, d_xmax, d_ymax = tuple(bbox2)
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)
    intersection = max(0, xb - xa) * max(0, yb - ya)
    return intersection


def compute_tot_intersection(bboxes):
    """
    Calculating the intersection area of all bounding boxes

    :param bboxes: box coordinates for all predicted defects
    :return: total area of intersection between all boxes
    """
    all_combinations = itertools.combinations(bboxes, 2)
    intersections = [compute_intersection(*combination) for combination in all_combinations]
    tot_area = sum(intersections)
    return tot_area


def isboxinside(bbox1, bbox2):
    """
    Function to remove overlapping bounding boxes

    :param bbox1: Box coordinates (xmin, ymin, xmax, ymax)
    :param bbox2: Box coordinates (xmin, ymin, xmax, ymax)
    :return: flag value
    """
    cutoff = 0.8
    g_xmin, g_ymin, g_xmax, g_ymax = tuple(bbox1)
    d_xmin, d_ymin, d_xmax, d_ymax = tuple(bbox2)
    flag = False
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)
    intersection = max(0, xb - xa) * max(0, yb - ya)
    area = max(0, g_xmax - g_xmin) * max(0, g_ymax - g_ymin)
    if intersection/area >=cutoff:
        flag = True
    return flag
  
  
def adjust_gamma(image, gamma=1.35):
    """
    Function for gamma correction

    :param image: input image
    :param gamma: gamma value to add to image
    :return: augumented image
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
    
  
def obj_helper(report_dict, im, draw, image, defect_c, all_boxes, cnt):
    """
    Helper function to calculate total_defective_area of predicted bboxes and saving predicted images to resp. location

    :param report_dict: dict to append values to after total area calculation and setting some flags.
    :param im: image with predicted boxes and caption which will be saved
    :param draw: image with no boxes or caption( used for unknown images)
    :param image: image name
    :param defect_c: list of defects present in image
    :param output_folder: folder location to save images
    :param all_boxes: bbox coordinates for a image

    :return: updated report_dict with total_defective_area and some other keys.
    """
    new_name = "_".join(defect_c)
    total_defective_area = 0

    if len(defect_c) > 0:              # update total defective area, has_unknown for defect images
        try:
            logger.info("Calculating total area for all the defects and average confidence: " + new_name + "_" + image)
            diction_key_list = report_dict[image]["defects"].keys()
            for defect_key in diction_key_list:

                report_dict[image]["defects"][defect_key]["confidence"] = round(report_dict[image]["defects"][defect_key]\
                ["confidence"]/report_dict[image]["defects"][defect_key]["count"], 4)

                report_dict[image]["total_defective_area"] = total_defective_area + report_dict[image]["defects"]\
                    [defect_key]["area"]

                total_defective_area = report_dict[image]["total_defective_area"]
            report_dict[image]["total_defective_area"] -= compute_tot_intersection(all_boxes)
            report_dict[image]["has_unknown"] = False
        except Exception:
            logger.error("Error in calculating total area for all the defects and average confidence: " + new_name+"_" \
                         + image, exc_info=True, stack_info=True)

    else:                         # update all dict details for unknown images and saving it
        try:
            report_dict[image]["total_defective_area"] = 0.0
            report_dict[image]["has_unknown"] = True
            report_dict[image]["defect_index"] = 90.0
            cnt += 1
        except Exception:
            logger.error("Error in saving Unknown image detection result: " + 'unknown' + "_" + image, exc_info=True,\
                         stack_info=True)

    return report_dict, cnt, im



def obj_detection(image, model, input_folder):
    """
    Localizing the defects in images

    :param image: list of images classified as defective by classification model
    :param input_folder: input folder path

    :return: updated report_dict with detection results

    """
    # Read the graph.
    logger.info("Object detection model started")
    
    cnt = 0  # unknown image count tracker

    report_dict = {}

    print('image', image)
    try:
        logger.info("reading images for object detection")
        img = read_image_bgr(os.path.join(input_folder, image))
    except Exception:
        logger.error("Error in loading the image: " + image, exc_info=True, stack_info=True)

    report_dict[image] = {"defects": {}}
    
    defect_c = set()
    bboxes = defaultdict(list)

    # copy to draw on
    draw = img.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # add gamma corection
    logger.info("Adding Gamma Correction")
    img = adjust_gamma(img)

    # preprocess image for network
    logger.info("Image Pre Processing")
    img = preprocess_image(img)
    img, scale = resize_image(img, 600, 1000)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.22:
            break
            
        # remove overlap boxes
        bboxes_list = [list(box) for box in boxes[0]]
        box_test = list(box)

        boxes_to_test = [b for b in bboxes_list if b != box_test and list(scores[0])[list(bboxes_list).index(b)] >= 0.22 \
                        and list(labels[0])[list(bboxes_list).index(b)] == label]
        test_flags = [isboxinside(box_test, b) for b in boxes_to_test]

        if any(test_flags):
            continue

        defect_class = labels_to_names[label]
        
        # start - code to remove all the edge tears boxes
        #if defect_class == 'tear' and (box[0] > 880 or box[2] < 180) :
        # continue
        # end
        defect_c.add(defect_class)
        bboxes[defect_class].append([box[0], box[1], box[2], box[3]])

        score *= 100
        
        try:
            logger.info("Appending existing value for: " + image)
            report_dict[image]["defects"].update({defect_class: {"area": compute_bbox_area(box) + report_dict[image] \
                ["defects"][defect_class]["area"], "confidence": score + report_dict[image]["defects"][defect_class] \
                ["confidence"], "count": 1 + report_dict[image]["defects"][defect_class]["count"]}})
        except KeyError:
            logger.info("Creating Dictionary for : " + image)
            report_dict[image]["defects"].update({defect_class: {"area": compute_bbox_area(box), "confidence": score,\
                                                                "count": 1}})
        try:
            logger.info("creating bounding boxes for : " + image)
            b = box.astype(int)
            b = np.array(b).astype(int)
            im = cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 2, cv2.LINE_AA)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            b = np.array(box).astype(int)
            cv2.putText(im, caption, (b[0] + 2, b[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(im, caption, (b[0] + 2, b[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception:
            logger.error("Error in bounding box creation: " + image, exc_info=True, stack_info=True)

    try:
        all_bboxes = [bboxes[d] for d in defect_c]
        all_boxes = list(itertools.chain.from_iterable(all_bboxes))
        # report_dict, cnt = obj_helper(report_dict, im, draw, image, defect_c, output_folder, all_boxes, cnt)

        logger.info("Updating Area and defct index")
        if defect_c:                  # update defect area, total area, defect index for known defects
            report_dict, cnt, im = obj_helper(report_dict, im, draw, image, defect_c, all_boxes, cnt)
            for d in defect_c:
                report_dict[image]["defects"][d]["area"] = round(((report_dict[image]["defects"][d]["area"]\
                                - compute_tot_intersection(bboxes[d])) / (draw.shape[0] * draw.shape[1])) * 100, 4)

            report_dict[image]["total_defective_area"] = round((report_dict[image]["total_defective_area"]\
                                                                    / (draw.shape[0] * draw.shape[1])) * 100, 4)

            # calculate defect index
            diction_key_list = report_dict[image]["defects"].keys()
            area_per_defect = [report_dict[image]["defects"][defect]["area"] for defect in diction_key_list]
            weighted_area_per_defect = [w * a for w, a in zip([weights[d] for d in diction_key_list],\
                                                            area_per_defect)]
            report_dict[image]["defect_index"] = round((100 - min(sum(weighted_area_per_defect), 100)), 2)
        else:
            report_dict, cnt, im = obj_helper(report_dict, draw, draw, image, defect_c, all_boxes, cnt)

    except Exception:
        logger.error("Error in Obj detection helper", exc_info=True, stack_info=True)

    return report_dict, im
