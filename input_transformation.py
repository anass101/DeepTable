import numpy as np
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import pickle 


def read_data(inp):
    with open(inp,'rb') as f: 
        [X,y] = pickle.load(f)
    return [X,y]


def transform_tables(inp, config):

	MAX_COL=9
	MAX_COL_LENGTH=9
	MAX_CELL_LENGTH=4
	

	[X,y] = read_data(inp)
	
	texts = ["XXX"] + [' '.join(text_to_word_sequence(' '.join(sum(x,[])),lower=True)) for x in X]
	
	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	
	

	if config == "train":
	
		data = np.zeros((int(len(X)*0.75), MAX_COL, MAX_COL_LENGTH,MAX_CELL_LENGTH), dtype='int32')

		X = X[0:int(len(X)*0.75)]
		y = y[0:int(len(y)*0.75)]

	else:
		data = np.zeros((len(X)-int(len(X)*0.75), MAX_COL, MAX_COL_LENGTH,MAX_CELL_LENGTH), dtype='int32')
	
		X = X[int(len(X)*0.75):]
		y = y[int(len(y)*0.75):]
		
	for i, table in enumerate(X):
		for j, col in enumerate(table):
			if j< MAX_COL:
				for k, cell in enumerate(col):
					if k<MAX_COL_LENGTH:
						z=0
						for _,word in enumerate(text_to_word_sequence(cell,lower=True)):	
							if z<MAX_CELL_LENGTH:
								if tokenizer.word_index.get(word) is not None:
									data[i,j,k,z] = tokenizer.word_index[word]
									z=z+1                    
	
	return data, np.array(y),tokenizer.word_index
	
