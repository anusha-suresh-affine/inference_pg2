import os
import datetime

def get_images(timestamp):
	input_folder = 'D:\\PG_Germany\\Input'
	return os.listdir(input_folder)

def store_image(details):
	pass

def check_if_created_before(ctime):
	if ctime > datetime.datetime.now() - datetime.timedelta(seconds = 15):
		return True
	else:
		return False
