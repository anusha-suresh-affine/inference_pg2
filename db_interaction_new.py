from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, MetaData, Column, String, Table
from sqlalchemy.orm import Session
from configuration import *
import pymysql
import os

import logging
logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(message)s')
fh = logging.FileHandler('db_interaction.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
os.environ['TZ'] = 'Europe/Berlin'

Base = automap_base()

# dbPath = 'C:\\Users\\affine\\Desktop\\pg__phase2-master\\pg__phase2-master\\pg_phase2\\db.sqlite3'
# engine = create_engine('sqlite:///%s' % dbPath, echo=True)

def new_connection():
	return create_engine('mysql+pymysql://' + 'affine:affine123$' + '@127.0.0.1:3306/'+'pg_realtime')

engine = new_connection()

metadata = MetaData(engine)
Base = automap_base()
Base.prepare(engine, reflect=True)

session = Session(engine)

Camera = Base.classes.interface_camera
Product = Base.classes.interface_product
ProductCamera = Base.classes.interface_productcamera
Image = Base.classes.interface_image
ImageClassification = Base.classes.interface_imageclassification
Defect = Base.classes.interface_defect



def save_details(table, details):
	logger.info('Saving new image details. ' + str(details))
	new_row = table()
	for key, value in details.items():
		setattr(new_row, key, value)
	session.add(new_row)
	try:
		session.commit()
	except Exception as e:
		logger.error(str(e))
		session.rollback()




def query_filter(table, arguments):
	logger.info('Querying ' + table + 'for ' + arguments)
	return session.query(table).filter_by(**arguments)


def query_last(table, arguments):
	logger.info('Querying ' + table + 'for ' + arguments)
	queries = session.query(table).filter_by(**arguments)
	return queries[-1]

def query_first(table, arguments):
	logger.info('Querying ' + table + 'for ' + arguments)
	return session.query(table).filter_by(**arguments).first()
