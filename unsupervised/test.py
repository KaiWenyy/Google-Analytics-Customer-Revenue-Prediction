"""An example to get each feature in data."""


import importlib
import pdb
import numpy as np 

import preproc
import utils
import clustering
from sklearn.preprocessing import StandardScaler


def collect_features(df):
	channelGrouping = utils.get_one_feature(df, "channelGrouping", to_onehot=True)  # One-hot string
	date = utils.get_one_feature(df, "date")  # Need processing, [20171016 20171016 20171016]
	browser = utils.get_one_feature(df, "device", "browser", to_onehot=True)  # One-hot string, too many classes?
	isMobile = utils.get_one_feature(df, "device", "isMobile")  # True, False
	fullVisitorId = utils.get_one_feature(df, "fullVisitorId")  # Already long int.
	
	continent = utils.get_one_feature(df, "geoNetwork", "continent", to_onehot=True)  # One-hot string
	subContinent = utils.get_one_feature(df, "geoNetwork", "subContinent", to_onehot=True)  # One-hot string with space
	country = utils.get_one_feature(df, "geoNetwork", "country", to_onehot=True)  # One-hot string with space

	hitNumber = utils.get_hits_feature(df, "hitNumber", to_float=True)  # A list, from 1 to hits, [1  2  3  4  5  6  7  8  9]
	time = utils.get_hits_feature(df, "time", to_float=True)  # [48297   65863   90876  104642  238657  238660  256359  261292]
	hour = utils.get_hits_feature(df, "hour", to_float=True)  # [6  6  7  7  7  7  7  7  7  7  7  7  7  7  7]
	minute = utils.get_hits_feature(df, "minute", to_float=True)  # [56 57  0  3  4  5  6  6  7  7 11 12 12 14 14]
	isInteraction = utils.get_hits_feature(df, "isInteraction", to_float=True)  # [1 1 1 1 1 1 1 1 1]
	#isEntrance = utils.get_hits_feature(df, "isEntrance", to_float=True)  # KeyError
	#isExit = utils.get_hits_feature(df, "isExit", to_float=True)  # KeyError
	# Many other information here.

	# 'socialEngagementType'
	visits = utils.get_one_feature(df, "totals", "visits", to_float=True)  # Should be int. [1, 1, 1]. All are ones?
	hits = utils.get_one_feature(df, "totals", "hits", to_float=True)  # Should be int. [9, 15, 15]
	pageviews = utils.get_one_feature(df, "totals", "pageviews", to_float=True)  # Should be int. [9, 12, 15]
	#bounces = utils.get_one_feature(df, "totals", "bounces")  # KeyError
	#newVisits = utils.get_one_feature(df, "totals", "newVisits")  # KeyError
	#sessionQualityDim = utils.get_one_feature(df, "totals", "sessionQualityDim")  # KeyError

	#source = utils.get_one_feature(df, "trafficSource", "source")  # SyntaxError: invalid syntax?
	#keyword = utils.get_one_feature(df, "trafficSource", "keyword")  # KeyError
	visitNumber = utils.get_one_feature(df, "visitNumber")  # Already int. [4, 11, 6]
	
	x = utils.concatenate_all_features([channelGrouping, date, browser, isMobile, continent, visits, hits, pageviews, visitNumber])
	return x


def main():
	df, allVisitorID = preproc.read_data()
	x = collect_features(df)
	y = utils.get_target(df, allVisitorID)  # Dict: fullVisitorID to target
	pdb.set_trace()

	
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	clustering.apply_kmeans(x, x)
	clustering.apply_hierarchical(x, x)

	"""TODO:
	1. Use hits information (time series) to do feature engineering.
	2. Use user information. Same user should has same user embedding.
	3. More data.
	4. What is demo dataset?
	"""


if __name__ == '__main__':
	main()

