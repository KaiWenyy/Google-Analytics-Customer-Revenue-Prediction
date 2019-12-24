import numpy as np 
import sys
import pdb
import importlib
import utils

from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input, Dense, Concatenate, Activation, Dropout, Embedding, Flatten
from tensorflow.compat.v1.keras.layers import LSTM
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import optimizers
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)
K.set_session(session)


def slice(x, i):
	return x[:, i:i+1]


def build_model(units, inputs_dim, output="regression", sparse_dim=[], with_ts=False, ts_maxlen=0):
	assert output == "regression" or output == "binary_clf", "This output type is not supported."
	assert len(sparse_dim) == inputs_dim[1], "Dimension not match."
	
	# Inputs for basic features.
	inputs1 = Input(shape=(inputs_dim[0],), name="basic_input")
	x1 = Dense(units, kernel_regularizer='l2', activation="relu")(inputs1)
	
	# Inputs for long one-hot features.
	inputs2 = Input(shape=(inputs_dim[1],), name="one_hot_input")
	for i in range(len(sparse_dim)):
		if i == 0:
			x2 = Embedding(sparse_dim[i], units, mask_zero=True)(slice(inputs2, i))
		else:
			tmp = Embedding(sparse_dim[i], units, mask_zero=True)(slice(inputs2, i))
			x2 = Concatenate()([x2, tmp])
	x2 = tf.reshape(x2, [-1, units*inputs_dim[1]])
	x = Concatenate()([x1, x2])

	if with_ts:
		inputs3 = Input(shape = (None, inputs_dim[2],), name="ts_input")
		x3 = LSTM(units, input_shape=(ts_maxlen, inputs_dim[2]), return_sequences=0)(inputs3)	
		x = Concatenate()([x, x3])
	
	x = Dense(units, kernel_regularizer='l2', activation = "relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(units, kernel_regularizer='l2', activation = "relu")(x)
	x = Dropout(0.5)(x)

	if output == "regression":
		x = Dense(1, kernel_regularizer='l2')(x)
		model = Model(inputs=[inputs1, inputs2], outputs=x)
		if with_ts: model = Model(inputs=[inputs1, inputs2, inputs3], outputs=x)
		model.compile(optimizer='adam', loss='mean_squared_error')

	elif output == "binary_clf":
		x = Dense(1, kernel_regularizer='l2', activation="sigmoid")(x)
		model = Model(inputs = [inputs1, inputs2], outputs=x)
		if with_ts: model = Model(inputs=[inputs1, inputs2, inputs3], outputs=x)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

	#model.summary()
	return model


def evaluate_f1(pred, real):
	pred = np.where(pred > 0.5, 1.0, 0.0)
	print('check labels =', np.mean(real), np.mean(pred))
	print('real =', real.shape, real[20:30], np.mean(real))
	print('pred =', pred.shape, pred[20:30], np.mean(pred))

	# f1 score
	f1 = precision_recall_fscore_support(real, pred, average='micro')
	print('f1 =', [round(e, 2) for e in f1[:3]], f1[3])
	f1 = precision_recall_fscore_support(real, pred, average='macro')
	print('f1 =', [round(e, 2) for e in f1[:3]], f1[3])
	#print('f1 =', precision_recall_fscore_support(real, pred))

	# AUC
	#fpr, tpr, thresholds = metrics.roc_curve(real, pred, pos_label=2)
	#print('AUC =', metrics.auc(fpr, tpr))
	#print('fpr, tpr, thresholds =', fpr, tpr, thresholds)
	

# usage: time python3 baseline.py 
def main():
	# Global settings
	with_ts = True  # If ts data is ready.
	train_ratio = 0.9
	ts_maxlen = 30  # Here: Max length of time series data for each sample.
	units = 64 
	
	# Load data
	path = "../test_train/npy_file/"
	x1 = np.load(path + "x1_train.npy")
	x2 = np.load(path + "x2_train.npy")[:, :]
	y = np.load(path + "train_y.npy")
	x1_test = np.load(path + "x1_test.npy")
	x2_test = np.load(path + "x2_test.npy")[:, :]
	if with_ts: 
		x3 = np.load(path + "x3_train.npy")  
		x3_test = np.load(path + "x3_test.npy")
	if with_ts: x1, x2, x3, y = shuffle(x1, x2, x3, y)  
	else: x1, x2, y = shuffle(x1, x2, y)

	# Set parameters from data.
	y = np.log1p(y)
	original_test_size = x1_test.shape[0]
	inputs_dim = [x1.shape[1], x2.shape[1]]  # Here: you should specify the dimension of x1 and x2.
	sparse_dim = [int(np.max(x2[:, i])) + 1 for i in range(x2.shape[1])]
	print("sparse_dim =", sparse_dim)
	print("Check x1:", [np.max(x1[:, i]) for i in range(x1.shape[1])])
	

	# Train/valid split
	train_num = int(y.shape[0] * train_ratio)
	x1_train, x1_valid = x1[:train_num], x1[train_num:]
	x2_train, x2_valid = x2[:train_num], x2[train_num:]
	if with_ts: x3_train, x3_valid = x3[:train_num], x3[train_num:]
	y_train, y_valid = y[:train_num], y[train_num:]
	print('Check shape:', x1_train.shape, x1_valid.shape, y_train.shape, y_valid.shape)
	
	# Training binary clf
	y_train_binary = np.where(y_train > 0, 1.0, 0.0)
	y_valid_binary = np.where(y_valid > 0, 1.0, 0.0)
	if with_ts:
		x_train = [x1_train, x2_train, x3_train]
		x_valid = [x1_valid, x2_valid, x3_valid]
		x_test = [x1_test, x2_test, x3_test]
	else:
		x_train = [x1_train, x2_train]
		x_valid = [x1_valid, x2_valid]
		x_test = [x1_test, x2_test]
	
	class_weight = {0: 1., 1: 100.}
	model = build_model(units, inputs_dim, output="binary_clf", sparse_dim=sparse_dim, ts_maxlen=ts_maxlen)
	model.fit(x_train, y_train_binary, epochs=3, batch_size=256, 
				class_weight=class_weight,
				validation_data=(x_valid, y_valid_binary))
	print('Check acc =', y_train_binary[20:30], model.predict(x_train)[20:30, 0])
	
	evaluate_f1(model.predict(x_train)[:, 0], y_train_binary)
	evaluate_f1(model.predict(x_valid)[:, 0], y_valid_binary)
	baseline_pred = model.predict(x_valid)[:, 0]
	baseline_pred *= 0
	baseline_pred[0] = 1
	evaluate_f1(baseline_pred, y_valid_binary)
	
	# If clf predicts "buy", then select these data.
	x_train, y_train, index = utils.get_buy_data(x_train + [y_train], model, with_ts)
	x_valid, y_valid, index = utils.get_buy_data(x_valid + [y_valid], model, with_ts)
	x_test, test_index = utils.get_buy_data([x1_test, x2_test], model, with_ts, with_y=False)
	
	# Train regression
	model = build_model(units, inputs_dim, output="regression", sparse_dim=sparse_dim, ts_maxlen=ts_maxlen)
	model.fit(x_train, y_train, epochs=30, batch_size=256, validation_data=(x_valid, y_valid))
	print('Check =', y_train[:10], model.predict(x_train)[:10, 0])

	# Get final testing results.
	pdb.set_trace()
	pred = model.predict(x_test)[:, 0]
	result = utils.combine_test_result(test_index, pred, original_test_size)
	write_result(result)


if __name__ == '__main__':
	main()

