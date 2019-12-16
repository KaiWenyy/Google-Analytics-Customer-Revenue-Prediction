import preproc
import importlib
import pdb
import numpy as np 

df, allVisitorID = preproc.read_data()
x = preproc.get_one_feature(df, "totals", "hits", to_float=True)
y = preproc.get_target(df, allVisitorID)
pdb.set_trace()
