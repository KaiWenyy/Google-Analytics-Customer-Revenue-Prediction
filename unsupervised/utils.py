import numpy as np 
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


def get_target(df, allVisitorId):
    target = {}
    for visitor in allVisitorId:
        target[visitor] = 0
    for index in df.index:
        visitor = df.at[index, "fullVisitorId"]
        target[visitor] += float(df.at[index, "totals"]["transactionRevenue"])
    for i, visitor in enumerate(allVisitorId):
        target[visitor] = np.log(target[visitor] + 1)
        if i < 3:
            print("VisitorID = %d, target = %.2f" % (visitor, target[visitor]))
    return target


def get_one_feature(df, column="totals", subcolumn=None, to_float=False, to_onehot=False):
    assert to_float + to_onehot < 2, "Can only transform to one of the types."
    if subcolumn != None:
        data = df[column]
        container = [d[subcolumn] for d in data]
    else:
        container = df[column]
        # df.channelGrouping.unique()
    if to_float:
        container = [float(n) for n in container]
    
    container = np.array(container).reshape([-1, 1])
    if to_onehot:
        container = one_hot_encoder(container)
    print(column, subcolumn, container.shape, container[:3])
    return container


def get_hits_feature(df, subcolumn, to_float=False):
    """Features in "hits":
        'hitNumber', 'time', 'hour', 'minute', 'isInteraction', 'isEntrance', 'page', 'transaction', 
        'item', 'appInfo', 'exceptionInfo', 'product', 'promotion', 'eCommerceAction', 'experiment', 
        'customVariables', 'customDimensions', 'customMetrics', 'type', 'social', 'contentGroup', 
        'dataSource', 'publisher_infos'
    """
    data = df["hits"]  
    container = []
    if to_float:
        for d in data:  # "d" is a list containing several dicts without fixed length.
            container.append([float(e[subcolumn]) for e in d])
    else:
        for d in data:
            container.append([d[h][subcolumn] for h in len(d)])
    container = pad_sequences(container, padding="pre", value=0.0, maxlen=30)  # Pad all lists to fixed length.
    # TODO: Improve how to deal with len > maxlen.
    print(subcolumn, container.shape, container[:3])
    return container


def one_hot_encoder(arr):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(arr)
    print("Categories:", len(enc.categories_[0]), enc.categories_[0][:10])
    arr = enc.transform(arr).toarray()
    return arr


def concatenate_all_features(features_list):
    assert len(features_list) > 0, "Empty features list."
    print(features_list[0].shape)
    x = np.empty([features_list[0].shape[0], 0])
    print(x.shape)
    for feature in features_list:
        x = np.concatenate([x, feature], axis=1)
    print("X.shape =", x.shape)
    return x




