import os
print(os.getcwd())
import time
import datetime
import cv2
from load_model import load_model
from historian import get_images, store_image
from db_interaction import *
from classification import classification
from objectdet import obj_detection
#load classification model
print('Loading classification model')
classify = load_model('classification')
#load object detection model
print('Loading object det model')
detect = load_model('object_detect')
# cwd = os.getcwd()
output_images = 'output_images'


def get_pc_id(image_name):
	camera_name = image_name.split('_')[0]
	print('Querying for camera_name: ' + camera_name)
	camera = query_filter(Camera, {'name': camera_name})
	camera = camera[0]
	pc = query_last(ProductCamera, {'camera_id': camera.id})
	return pc.id


def save_image_details(image_name):
	pc = get_pc_id(image_name)
	image_details = {
		'name': image_name,
		'time': datetime.datetime.now(),
		'product_camera_id': pc
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
						if not k == 'confidence': 
							defect_results[key+'_'+k] = v
						else:
							defect_results[key+'_'+k] = float(v)
			else:
				if field == 'total_defective_area':
					defect_results[field] = float(result[field])
				else:
					defect_results[field] = result[field]
		defect_id = save_details(Defect, defect_results)
	except Exception as e:
		print(str(e))
		import pdb; pdb.set_trace()


#start
while(1):
	print('starting iteration')
	timestamp_start = datetime.datetime.now()
	#query ge historian 
	images = get_images('eneno')
	if images:
		for image in images:
			try:
				# import pdb; pdb.set_trace()
				save_image_details(image)
				image_stored = query_last(Image,{'name': image})
				# import pdb; pdb.set_trace()
				input_folder = 'C:\\Users\\anusha\\input'
				classification_results = classification(image, classify, input_folder)
				save_classification_results(classification_results[image], image_stored.id)
				if classification_results[image]['is_defective']:
					print('defective image processing')
					obj_det_result,img = obj_detection(image, detect, input_folder)
					cv2.imwrite(os.path.join(output_images, image), img)
					obj_det_result[image]['image_path'] = os.path.join(os.getcwd(), 'output_images', image)
					save_defect_results(obj_det_result[image], image_stored.id)
					# img write to ge historian
			except Exception as e:
				print('ERRORRRRRRRRRRRRR!!')
				print(str(e))
	if timestamp_start + datetime.timedelta(seconds=15) > datetime.datetime.now():
		seconds = (timestamp_start + datetime.timedelta(seconds=15) - datetime.datetime.now()).total_seconds()
		print('waiting ' + str(seconds))
		if seconds>0:
			time.sleep(seconds)
	print('one iteration done')
	break







