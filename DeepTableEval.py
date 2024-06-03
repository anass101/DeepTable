import argparse
import pandas as pd
import numpy as np
import pickle
import os

from keras.models import load_model 
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm

from input_transformation import transform_tables

os.environ['KERAS_BACKEND']='tensorflow'
np.random.seed(813306)
	
if __name__ == "__main__":

	parser  = argparse.ArgumentParser()
	parser.add_argument("-m", "--model_path", type=str)
	parser.add_argument("-d", "--data_file", type=str)
	parser.add_argument("-o", "--output_path", type=str)
	args = parser.parse_args()

	model_path = args.model_path
	data_file = args.data_file
	output_path = args.output_path

	# variable initialization
	MAX_COL=9
	MAX_COL_LENGTH=9
	MAX_CELL_LENGTH=4
	X_test,y_test,dictionary = transform_tables(data_file, "test")
	
	# load model
	model = load_model(model_path)
	
	# predict labels
	pred = model.predict(X_test, verbose=1)

	
	refs = [r.tolist().index(max(r.tolist())) for r in y_test]
	preds=[p.tolist().index(max(p.tolist())) for p in pred]
	
	# write predictions in a file
	refs_preds = pd.DataFrame([(r.tolist().index(max(r.tolist())),p.tolist().index(max(p.tolist()))) for r,p in zip(y_test,pred)], columns = ["reference","prediction"])
	refs_preds.to_csv(output_path+".csv",index=False)
	
	# display performances
	print(cr(refs,preds,digits = 4))
	print("confusion_matrix:\n",cm(refs,preds))	
	print("predictions are saved in \""+output_path+".csv\"")