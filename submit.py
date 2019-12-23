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

def np_array_to_df(array):
	df = pd.DataFrame(data=array, #[1:,1:],    # values  index=array[1:,0],    # 1st column as index              	 
              	 columns=['fullVisitorId','PredictedLogRevenue'])
	return df

def produce_submission(pred_df, sample_df):  # assume predict dataframe : fullVisitorId, revenue
	#sub_df = pd.DataFrame({"fullVisitorId":test_id})
	#pred_test[pred_test<0] = 0
	#df["PredictedLogRevenue"] = np.expm1(pred_test)
	pred_df = pred_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
	#df.columns = ["fullVisitorId", "PredictedLogRevenue"]
	pred_df["PredictedLogRevenue"] = np.log1p(pred_df["PredictedLogRevenue"])
	for i in pred_df.index:
		ind = sample_df['fullVisitorId'][ sample_df['fullVisitorId']==pred_df['fullVisitorId'][i] ].index
		sample_df['PredictedLogRevenue'][ind] = pred_df['PredictedLogRevenue'][i]
	sample_df.to_csv("baseline.csv", index=False)

def match(sample_df, pred_df):
	diff1 = list( set(sample_df['fullVisitorId']) - set(pred_df['fullVisitorId'].unique()) )
	diff2 = list( set(pred_df['fullVisitorId'].unique()) - set(sample_df['fullVisitorId']) )

	print(diff1)
	print(diff2) 


if __name__ == '__main__':
	#test = load_other_df('test/others/test_df2.csv')
	sample = load_submission('sample_submission_v2.csv')
	pred = np_array_to_df(array)
	#match(sample, test)
	produce_submission(pred, sample)