from inputvars import *
md1,output,inp = inputvarsEval(sys.argv[1:],sys.argv[0])
import pandas as pd
import numpy as np
import pickle
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import load_model 
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm
np.random.seed(813306)	
from input_transformation import *

def read_input(inp):
	with open(inp,'rb') as f:
		[data,Y,dicTab]=pickle.load(f)
	train_size = int(Y.shape[0]*0.75)
	return data[train_size:,:,:,:],Y[train_size:]

	
if __name__ == "__main__":
	
	# variable initialization
	MAX_COL=9
	MAX_COL_LENGTH=9
	MAX_CELL_LENGTH=4
	X_test,y_test,dictionary = transform_tables(inp, "test")
	
	# load model
	model = load_model(md1)
	
	# predict labels
	pred = model.predict(X_test, verbose=1)

	
	refs = [r.tolist().index(max(r.tolist())) for r in y_test]
	preds=[p.tolist().index(max(p.tolist())) for p in pred]
	
	# write predictions in a file
	refs_preds = pd.DataFrame([(r.tolist().index(max(r.tolist())),p.tolist().index(max(p.tolist()))) for r,p in zip(y_test,pred)], columns = ["reference","prediction"])
	refs_preds.to_csv(output+".csv",index=False)
	
	# display performances
	print(cr(refs,preds,digits = 4))
	print("confusion_matrix:\n",cm(refs,preds))	
	print("predictions are saved in \""+output+".csv\"")