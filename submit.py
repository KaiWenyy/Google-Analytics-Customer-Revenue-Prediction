import pandas as pd 
from newproc import load_other_df, load_hits_df, intersection
import numpy as np


def load_submission(csv_name, nrows=None):
	print('load file '+ csv_name+' ...')
	df = pd.read_csv(csv_name,
                     dtype = {'fullVisitorId': 'str', 'visitId': 'str'},
                     nrows = nrows)
	return df

def read_npy(filename):
	return df

def np_array_to_df(array,test_df):
	a = np.c_[test['fullVisitorId'].values, array]
	print(a.shape)
	df = pd.DataFrame(data=a, #[1:,1:],    # values  index=array[1:,0],    # 1st column as index              	 
              	 columns=['fullVisitorId','PredictedLogRevenue'])
	return df

def produce_submission(pred_df, sample_df, filename):  # assume predict dataframe : fullVisitorId, revenue
	#pred_test[pred_test<0] = 0
	#df["PredictedLogRevenue"] = np.expm1(pred_test)
	diff = match(sample_df, pred_df)
	print(len(diff))
	for i in diff:
		insert = pd.DataFrame([[i,0]], columns=['fullVisitorId','PredictedLogRevenue'])
		pred_df = pred_df.append(insert,ignore_index=True)
	pred_df = pred_df.groupby("fullVisitorId")['PredictedLogRevenue'].sum().reset_index()
	pred_df = pred_df.sort_values(by="fullVisitorId", ascending=True)
	pred_df['PredictedLogRevenue'] = np.log1p(pred_df['PredictedLogRevenue'])
	'''
	for i in pred_df.index:
		ind = sample_df['fullVisitorId'][ sample_df['fullVisitorId']==pred_df['fullVisitorId'][i] ].index
		sample_df['PredictedLogRevenue'][ind] = pred_df['PredictedLogRevenue'][i]
	'''

	#sample_df.to_csv("baseline.csv", index=False)
	pred_df.to_csv(filename, index=False)

def match(sample_df, pred_df):
	diff1 = list( set(sample_df['fullVisitorId']) - set(pred_df['fullVisitorId'].unique()) )
	#diff2 = list( set(pred_df['fullVisitorId'].unique()) - set(sample_df['fullVisitorId']) )
	return diff1 


if __name__ == '__main__':
	test = load_other_df('test/others/test_df2.csv')
	sample = load_submission('sample_submission_v2.csv')
	# one npy file
	npy = npyfilename # TODO
	filename = npy.split('.')[0] + "_baseline.csv"
	array = np.load(npy)
	pred = np_array_to_df(array[:,1], test)
	produce_submission(pred, sample, filename)

	'''
	# for many npy files
	files = ['result_full_data_model.npy', 'result_only_basic_data.npy', 'result_only_second_model.npy', 'result_only_ts_data.npy']
	for f in files:
		npy = 'results/'+ f
		filename = npy.split('.')[0] + "_baseline.csv"
		array = np.load(npy)
		pred = np_array_to_df(array[:,1], test)
		#match(sample, test)
		produce_submission(pred, sample, filename)
	'''