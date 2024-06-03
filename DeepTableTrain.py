import argparse
import os

import keras
from keras.callbacks import ModelCheckpoint

from input_transformation import transform_tables
from model_architecture import deep_table_model, read_embeddings

os.environ['KERAS_BACKEND']='tensorflow'

if __name__ == "__main__":
	
	parser  = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int)
	parser.add_argument("-lr", "--learning_rate", type=float)
	parser.add_argument("-emb", "--embed_loc", type=str)
	parser.add_argument("-d", "--data_file", type=str)
	parser.add_argument("-md", "--model_dir", type=str)
	args = parser.parse_args()

	epochs = args.epochs
	learning_rate = args.learning_rate
	embed_loc = args.embed_loc
	data_file = args.data_file
	model_dir = args.model_dir

	args.epochs
	# variable initialization
	MAX_COL=9
	MAX_COL_LENGTH=9	
	MAX_CELL_LENGTH=4	
	embedding_flag = 1
	learning_rate = float(learning_rate)
	epochs = int(epochs)
	filepath=model_dir+"/model"+"-{epoch:02d}-{val_loss:.4f}-{loss:.4f}.keras"
	inp = "C:/Users/Anass/Desktop/projects/DeepTable/tables.pickle"

	# read train samples
	X_train,y_train,dictionary = transform_tables(inp, "train")
	
	# read embedding vector
	embedding_matrix = read_embeddings(dictionary, embed_loc)
	
	# model initialization and training
	final_model = deep_table_model((MAX_COL,MAX_COL_LENGTH,MAX_CELL_LENGTH,),dictionary,embedding_matrix,embedding_flag)
	final_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=learning_rate), metrics=['accuracy'])
	callbacks_list = [ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min',save_freq = "epoch")]
	final_model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split =0.25, shuffle=True, callbacks=callbacks_list)
