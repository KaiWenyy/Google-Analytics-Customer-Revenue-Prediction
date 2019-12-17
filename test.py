"""An example to get each feature in data."""


import preproc
import importlib
import pdb
import numpy as np 


df, allVisitorID = preproc.read_data()
y = preproc.get_target(df, allVisitorID)  # Dict: fullVisitorID to target

channelGrouping = preproc.get_one_feature(df, "channelGrouping")  # One-hot string
date = preproc.get_one_feature(df, "date")  # Need processing, [20171016 20171016 20171016]

browser = preproc.get_one_feature(df, "device", "browser")  # One-hot string, too many classes?
isMobile = preproc.get_one_feature(df, "device", "isMobile")  # True, False

fullVisitorId = preproc.get_one_feature(df, "fullVisitorId")  # Already long int.
continent = preproc.get_one_feature(df, "geoNetwork", "continent")  # One-hot string
subContinent = preproc.get_one_feature(df, "geoNetwork", "subContinent")  # One-hot string with space
country = preproc.get_one_feature(df, "geoNetwork", "country")  # One-hot string with space

hitNumber = preproc.get_hits_feature(df, "hitNumber", to_float=True)  # A list, from 1 to hits, [1  2  3  4  5  6  7  8  9]
time = preproc.get_hits_feature(df, "time", to_float=True)  # [48297   65863   90876  104642  238657  238660  256359  261292]
hour = preproc.get_hits_feature(df, "hour", to_float=True)  # [6  6  7  7  7  7  7  7  7  7  7  7  7  7  7]
minute = preproc.get_hits_feature(df, "minute", to_float=True)  # [56 57  0  3  4  5  6  6  7  7 11 12 12 14 14]
isInteraction = preproc.get_hits_feature(df, "isInteraction", to_float=True)  # [1 1 1 1 1 1 1 1 1]
# Many other information here.

# 'socialEngagementType'
visits = preproc.get_one_feature(df, "totals", "visits", to_float=True)  # Should be int.
hits = preproc.get_one_feature(df, "totals", "hits", to_float=True)  # Should be int.
pageviews = preproc.get_one_feature(df, "totals", "pageviews", to_float=True)  # Should be int.

visitNumber = preproc.get_one_feature(df, "visitNumber")  # Already int.

