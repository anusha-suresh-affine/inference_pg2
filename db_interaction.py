from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, MetaData, Column, String, Table
from sqlalchemy.orm import Session

Base = automap_base()

# dbPath = 'C:\\Users\\anusha\\pg_phase2\\db.sqlite3'
# engine = create_engine('sqlite:///%s' % dbPath, echo=True)
engine = create_engine('mysql+pymysql://' + 'root:test@123' + '@127.0.0.1:3306/'+'pg_realtime')
    
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
	new_row = table()
	for key, value in details.items():
		setattr(new_row, key, value)
	session.add(new_row)
	session.commit()



def query_filter(table, arguments):
	return session.query(table).filter_by(**arguments)


def query_last(table, arguments):
	queries = session.query(table).filter_by(**arguments)
	return queries[-1]

def query_first(table, arguments):
	return session.query(table).filter_by(**arguments).first()



