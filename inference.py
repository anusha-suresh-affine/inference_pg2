import os
import time
import datetime
import cv2
import pdb
import logging

from load_model import load_model
from db_interaction import *
from classification import classification
from objectdet import obj_detection
from configuration import *

#load classification model
classify = load_model('classification')
#load object detection model
detect = load_model('object_detect')

logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(message)s')
fh = logging.FileHandler('inference.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
os.environ['TZ'] = 'Europe/Berlin'


def get_pc_id(image_name):
	camera_name = image_name.split('_')[0]
	logger.info('Querying for camera_name: ' + camera_name)
	camera = query_filter(Camera, {'name': camera_name})
	camera = camera[0]
	pc = query_last(ProductCamera, {'camera_id': camera.id})
	return pc.id


def save_image_details(image_name, image_path):
	pc = get_pc_id(image_name)
	image_details = {
		'name': image_name,
		'time': datetime.datetime.now(),
		'product_camera_id': pc,
		'image_path': image_path,
	}
	image_id = save_details(Image, image_details)
	return image_id


def save_classification_results(result, image_id):
	result_details = {
		'image_id': image_id,
		'is_defective': result['is_defective'],
		'confidence': float(result['confidence']),
		'time': datetime.datetime.now()
	}
	image_classification = save_details(ImageClassification, result_details)


def save_defect_results(result, image_id):
	try:
		defect_results = {}
		defect_results['image_id'] = image_id
		defect_types = ['contamination', 'tear', 'trim', 'wrinkle', 'mistracking']
		for field in result.keys():
			if field=='defects':
				for key in result[field].keys():
					for k, v in result[field][key].items():
						if k == 'confidence' or k == 'area':
							defect_results[key+'_'+k] = float(v)
						else:
							defect_results[key+'_'+k] = v
			else:
				if field == 'total_defective_area':
					defect_results[field] = float(result[field])
				else:
					defect_results[field] = result[field]
		defect_id = save_details(Defect, defect_results)
	except Exception as e:
		logger.error(str(e))


def get_images(image_list):
	date = datetime.datetime.now().strftime('%Y_%m_%d')
	try:
		dated_input = os.path.join(input_folder, date, input_folder_ext)
		logger.info('Reading images from ' + dated_input)
		start_delta = time.time()
		all_images = os.listdir(dated_input)
		delta_images = list(set(all_images) - set(image_list))
		delta_time = time.time() - start_delta
		logger.info('Delta took ' + str(delta_time))
		logger.info(str(delta_images))
		image_list = all_images
		return delta_images, dated_input, image_list
	except Exception as e:
		logger.error(str(e))
		return [], dated_input, image_list

def store_image(data, image_name, path):
	logger.info("Storing the defective image: " + image_name)
	cv2.imwrite(os.path.join(path, image_name), data)


#start
logger.info('Starting the persisitant loop')
try:
	today = datetime.datetime.now().strftime('%Y_%m_%d')
	read_folder = os.path.join(input_folder, today, input_folder_ext)
	image_list = os.listdir(read_folder)
except Exception as e:
	image_list = []
	logger.error(str(e))

while(1):
	logger.info('Starting an iteration')
	timestamp_start = time.time()
	logger.info('Getting images')
	images, image_path, image_list = get_images(image_list)
	logger.info('Found ' + str(len(images)) + ' images')
	if images:
		for image in images:
			try:
				logger.info('Saving image details')
				save_image_details(image, image_path)
				image_stored = query_last(Image,{'name': image})
				logger.info('The id of the saved image is: ' + str(image_stored.id))
				logger.info('Starting classification')
				classification_results = classification(image, classify, image_path)
				logger.info('Saving classification results')
				save_classification_results(classification_results[image], image_stored.id)
				if classification_results[image]['is_defective']:
					logger.info('Image was found to be defective. Starting object detection')
					obj_det_result,img = obj_detection(image, detect, image_path)
					store_image(img, image, output_folder)
					obj_det_result[image]['image_path'] = os.path.join(output_folder, image)
					logger.info('saving defects')
					save_defect_results(obj_det_result[image], image_stored.id)
					# to do: img write to ge historian
			except Exception as e:
				logger.error(str(e))
	if time.time() - timestamp_start < 15:
		seconds = 15 - (time.time() - timestamp_start)
		logger.info('waiting ' + str(seconds))
		if seconds > 0:
			time.sleep(seconds)
	logger.info('one iteration done')
