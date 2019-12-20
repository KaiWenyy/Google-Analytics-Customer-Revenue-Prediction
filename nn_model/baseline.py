import numpy as np 
import sys


from sklearn.utils import shuffle
#from sklearn import metrics
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Activation, Dropout
#from keras.layers import Lambda, Flatten, 
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras import optimizers

import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
#session = tf.Session(config=config)
#K.set_session(session)


def build_model(units, inputs_dim, output="regression", with_ts=False, ts_maxlen=0):
	assert output == "regression" or output == "binary_clf", "This output type is not supported."
	# TODO: Various units for x1 and x2.
	
	# Inputs for basic features.
	inputs1 = Input(shape=(inputs_dim[0],), name="basic_input")
	x1 = Dense(units, W_regularizer='l2', activation="relu")(inputs1)
	
	# Inputs for long one-hot features.
	inputs2 = Input(shape=(1,), name="one_hot_input")
	x2 = Embedding(inputs_dim[1], units, mask_zero=True)(inputs2)
	x = Concatenate()([x1, x2])

	if with_ts:
		inputs3 = Input(shape = (None, inputs_dim[2],), name="ts_input")
		x3 = LSTM(units, input_shape=(ts_maxlen, inputs_dim[2]), return_sequences=0)(inputs3)	
		x = Concatenate()([x, x3])
	
	x = Dense(units, W_regularizer = 'l2', activation = "relu")(x)
	x = Dropout(0.5)(x)

	if output == "regression":
		x = Dense(1, W_regularizer='l2')(x)
		model = Model(input = [inputs1, inputs2], output = x)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

	elif output == "binary_clf":
		x = Dense(1, W_regularizer='l2', activation="sigmoid")(x)
		model = Model(input = [inputs1, inputs2], output=x)
		model.compile(optimizer='adam', loss='mean_squared_error')

	model.summary()
	return model


def evaluate(model, x, real):
	pred = model.predict(x)[:, 0]
	pred = np.where(pred>0.5, 1, 0)
	print('check labels =', np.mean(real), np.mean(pred))
	print('real =', real.shape, real[:10], np.mean(real))
	print('pred =', pred.shape, pred[:10], np.mean(pred))

	# f1 score
	print('f1 =', precision_recall_fscore_support(real, pred, average='macro'))
	print('f1 =', precision_recall_fscore_support(real, pred))

	# AUC
	fpr, tpr, thresholds = metrics.roc_curve(real, pred, pos_label=2)
	print('AUC =', metrics.auc(fpr, tpr))
	print('fpr, tpr, thresholds =', fpr, tpr, thresholds)
	

# usage: time python3 baseline.py 
def main():
	# Global settings
	with_tf = False  # If ts data is ready.
	inputs_dim = [ ]  # Here: you should specify the dimension of x1 and x2.
	max_length = 0  # Here: Max length of time series data for each sample.
	units = 64 

	# Load data
	x1 = np.load("")
	x2 = np.load("")
	y = np.load("")
	x1_test = np.load("")
	x2_test = np.load("")
	if with_tf: 
		x3 = np.load("")  
		x3_test = np.load("")

	if with_tf:
		x1, x2, x3, y = shuffle(x1, x2, x3, y)  
	else:
		x1, x2, y = shuffle(x1, x2, y)
	
	# Train/valid split
	train_ratio = 0.2
	train_num = int(y.shape[0] * train_ratio)
	
	x1_train, x1_valid = x1[:train_num], x1[train_num:]
	x2_train, x2_valid = x2[:train_num], x2[train_num:]
	if with_tf:
		x3_train, x3_valid = x3[:train_num], x3[train_num:]
	y_train, y_valid = y[:train_num], y[train_num:]
	print('Check shape:', x1_train.shape, x1_valid.shape, y_train.shape, y_valid.shape)
	
	# Training binary clf
	model = build_model(units, inputs_dim, output="binary_clf", max_length=max_length)
	model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=256, validation_data=([x1_valid, x2_valid], y_valid))
	#print('Check acc =', y[:10], model.predict(x[:10])[:, 0])
	evaluate(model, [x1_train, x2_train], y_train)
	evaluate(model, [x1_valid, x2_valid], y_valid)

	# Train regression
	# If clf predicts "buy", then select these data.
	model = build_model(units, inputs_dim, output="regression", max_length=max_length)
	model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=256, validation_data=([x1_valid, x2_valid], y_valid))
	

if __name__ == '__main__':
	main()
