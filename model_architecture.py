import numpy as np
import gensim

import keras
from keras import regularizers
from keras.layers import Embedding,Dense, Input, Flatten, Embedding, Dropout,LSTM, Bidirectional, TimeDistributed,MaxPooling2D,Reshape,Reshape
from keras.models import Model 


def read_embeddings(dictionary, emb_file):
	
	modelemb = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=True)
	w2v = dict(zip(modelemb.index_to_key, modelemb.vectors))
	embedding_matrix = np.zeros((len(dictionary) + 1, 200))
	
	for j, i in dictionary.items():
		if w2v.get(j) is not None:
			embedding_matrix[i] = w2v[j]
			
	return embedding_matrix
	
def cell_encoder(input_shape,dic_length,embedding_matrix,embedding_flag):

    r_in = Input(shape=input_shape)
    if embedding_flag == 1:
        c_emb = Embedding(dic_length,200,weights=[embedding_matrix],input_length=input_shape[-1],trainable=True)(r_in)
    else:
        c_emb = Embedding(dic_length,200,input_length=input_shape[-1],trainable=True)(r_in)	
    c_lstm = Bidirectional(LSTM(50))(c_emb)
    c_dense = Dense(100,activation='relu')(c_lstm)
    c_dense = Dropout(0.1)(c_dense)
    c_dense = Dense(100,activation='relu')(c_dense)
    c_dense = Dropout(0.1)(c_dense)

    cell_model = Model(r_in, c_dense)
    print(cell_model.summary())

    return cell_model
	
def column_encoder(input_shape, config):

    input_layer = Input(shape = input_shape)
    if config == 1:
        conv_layer1 = MaxPooling2D((1,input_shape[1]))(input_layer)	
    else:
        conv_layer1 = MaxPooling2D((input_shape[0],1))(input_layer)
    dense_layer1 = Dense(100,activation='relu')(conv_layer1)
    dropout_layer1 = Dropout(0.1)(dense_layer1)
    dense_layer2 = Dense(100,activation='relu')(dropout_layer1)
    dropout_layer2 = Dropout(0.1)(dense_layer2)
    flat_layer1 = Flatten()(dropout_layer2)

    column_model = Model(input_layer,flat_layer1)
    print(column_model.summary())
    
    return column_model
	
def deep_table_model(input_shape,dictionary,embedding_matrix,embedding_flag):
	
    c_in = Input(shape=input_shape, dtype='float64')
    reshape_in = Reshape((input_shape[-2]*input_shape[-3],input_shape[-1]),input_shape = input_shape)(c_in)

    # column representation
    embedding_layer_col = TimeDistributed(cell_encoder((input_shape[-1],),len(dictionary)+1,embedding_matrix,embedding_flag))(reshape_in)
    col_in = Reshape((input_shape[-3],input_shape[-2],100),input_shape = (input_shape[-2]*input_shape[-3],100,))(embedding_layer_col)
    col_wise_layer = column_encoder((input_shape[-3],input_shape[-2],100,),1)(col_in)

    # row representation
    embedding_layer_row = TimeDistributed(cell_encoder((input_shape[-1],),len(dictionary)+1,embedding_matrix,embedding_flag))(reshape_in)
    row_in = Reshape((input_shape[-3],input_shape[-2],100),input_shape = (input_shape[-2]*input_shape[-3],100,))(embedding_layer_row)
    row_wise_layer = column_encoder((input_shape[-3],input_shape[-2],100,),0)(row_in)

    # concatenate rows and columns representations
    flats = keras.layers.Concatenate(axis=-1)([col_wise_layer,row_wise_layer])

    # softmax classifier
    final_dense = Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(flats)
        
    final_model = Model(c_in,final_dense)
    print(final_model.summary())

    return final_model
