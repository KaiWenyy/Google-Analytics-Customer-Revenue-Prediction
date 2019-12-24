import numpy as np 


def get_buy_data(x_list, model, with_tf, with_y=True, threshold=0.5):
	if with_tf:
		input_length = 3
	else:
		input_length = 2
	pred = model.predict(x_list[:input_length])[:, 0]
	#pdb.set_trace()
	print(pred.shape, len(list(pred)))

	index = [i for i, e in enumerate(list(pred)) if e > threshold]
	print("index =", len(index))
	#print(123)
	for i in range(len(x_list)):
		#print(x_list[i].shape)
		#print(x_list[i][index].shape)
		x_list[i] = x_list[i][index]
	
	if with_y:
		return x_list[:input_length], x_list[input_length], index
	else:
		return x_list[:input_length], index



def combine_test_result(buy_index, pred, original_n):
	result = []
	count = 0
	for i in range(original_n):
		if i in buy_index:
			result.append(pred[count])
			count += 1
		else:
			result.append(0)
	assert len(result) == original_n, "Length not match for testing results."
	return result


def write_result():
	# Not yet
	return 

