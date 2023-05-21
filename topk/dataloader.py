import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.utils import shuffle

import sys
sys.path.append("./topk/TreeBenchmark/")
sys.path.append("./topk/TreeBenchmark/python")

from python.model.encoder import Encoder

def load_data(data_path, dataset_name, seed=4, frac_train=0.8, convert_to_int=True, one_hot=True):
    path = os.path.join(data_path, "dataset_" + dataset_name + ".csv")
    data = open(path, "r").read().split('\n')[1:] # first row is column names
    
    # sometimes, last row is just a blank row
    if len(data[-1]) != data[-2]:
        data = data[:-1]
    
    data = [data[i].split(',') for i in range(len(data))]
    
    # order of shuffling
    np.random.seed(seed)
    order = np.arange(len(data))
    np.random.shuffle(order)
    
    num_train = int(len(data) * frac_train)
    
    # last column is class
    x_train = [data[index][0:-1] for index in order[:num_train]]
    y_train = [data[index][-1] for index in order[:num_train]]
    
    x_test = [data[index][0:-1] for index in order[num_train:]]
    y_test = [data[index][-1] for index in order[num_train:]]
    
    classes = sorted(list(set(y_train + y_test)))
    
    # make a dict of feature_id -> [possible string values for this feature]
    # feature_id ranges from {0, num_features-1}
    feature_value_dict = {}

    for i in range(len(x_train[0])): # iterate over all features
        vals = []
        for j in range(len(x_train)): # find all possible values of this feature
            vals.append(x_train[j][i])
        for j in range(len(x_test)):
            vals.append(x_test[j][i])
        distinct_vals = sorted(list(set(vals)))
        feature_value_dict[i] = distinct_vals
    
    num_features = len(feature_value_dict)
    # print(num_features)
    
    if convert_to_int:
        feature_string_to_int_mapper = {}
        for feature_id, feature_values in feature_value_dict.items():
            cnt = 0
            mapp = {}
            for value in feature_values:
                mapp[value] = cnt
                cnt += 1
            feature_string_to_int_mapper[feature_id] = mapp
        
        class_string_to_int_mapper = {}
        cnt = 0
        for c in classes:
            class_string_to_int_mapper[c] = cnt
            cnt += 1
            
        for i in range(len(x_train[0])):
            for j in range(len(x_train)):
                x_train[j][i] = feature_string_to_int_mapper[i][x_train[j][i]]
            for j in range(len(x_test)):
                x_test[j][i] = feature_string_to_int_mapper[i][x_test[j][i]]

        for j in range(len(y_train)):
            y_train[j] = class_string_to_int_mapper[y_train[j]]
        for j in range(len(y_test)):
            y_test[j] = class_string_to_int_mapper[y_test[j]]
            
        for feature_id, feature_values in feature_value_dict.items():
            feature_value_dict[feature_id] = sorted(list(feature_string_to_int_mapper[feature_id].values()))
            
        classes = sorted(list(class_string_to_int_mapper.values()))
        
    if convert_to_int and one_hot:
        enc = OneHotEncoder()
        enc.fit(x_train + x_test)

        x_train = enc.transform(x_train).toarray().tolist()
        x_test = enc.transform(x_test).toarray().tolist()
        
        feature_value_dict = {i: [0, 1] for i in range(len(x_train[0]))}
        
    return x_train, y_train, x_test, y_test, feature_value_dict, classes

def load_data_numerical(data_path, dataset_name, seed=4, frac_train=0.8, mode="custom-bucket", max_splits=200):
    path = os.path.join(data_path, "dataset_" + dataset_name + ".csv")
    dataframe = shuffle(pd.DataFrame(
        pd.read_csv(path)).dropna(), random_state=seed)
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape
    # print(m)
    
    if mode == "custom-bucket":
        num_buckets_per_feature = (max_splits // m) + 1
        encoder = Encoder(X.values[:,:], header=X.columns[:], 
                  mode="custom-bucketize", num_buckets_per_feature=num_buckets_per_feature)
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
    else:
        encoder = Encoder(X.values[:,:], header=X.columns[:])
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
    
    (n, z) = X.shape

    sample_size = int(frac_train * n )
    train_index = [ i for i in range(sample_size) ]
    test_index = [ i for i in range(sample_size, n) ]
    x_train, y_train = X.iloc[train_index].to_numpy().tolist(), y.iloc[train_index].to_numpy()[:, 0].tolist()
    x_test, y_test = X.iloc[test_index].to_numpy().tolist(), y.iloc[test_index].to_numpy()[:, 0].tolist()
    
    classes = sorted(list(set(y_train + y_test)))
    
    feature_value_dict = {}
    
    for i in range(len(x_train[0])): # iterate over all features
        vals = []
        for j in range(len(x_train)): # find all possible values of this feature
            vals.append(x_train[j][i])
        for j in range(len(x_test)):
            vals.append(x_test[j][i])
        distinct_vals = sorted(list(set(vals)))
        feature_value_dict[i] = distinct_vals
    
    num_features = len(feature_value_dict)
    
    class_string_to_int_mapper = {}
    cnt = 0
    for c in classes:
        class_string_to_int_mapper[c] = cnt
        cnt += 1
        
    for j in range(len(y_train)):
        y_train[j] = class_string_to_int_mapper[y_train[j]]
    for j in range(len(y_test)):
        y_test[j] = class_string_to_int_mapper[y_test[j]]
        
    classes = sorted(list(class_string_to_int_mapper.values()))
    
    return x_train, y_train, x_test, y_test, feature_value_dict, classes

def load_data_numerical_tt_split(data_path, dataset_name, seed=4, frac_train=0.8, mode="custom-bucket", max_splits=200):
    train_path = os.path.join(data_path, "dataset_" + dataset_name + "-train.csv")
    dataframe = pd.DataFrame(
        pd.read_csv(train_path)).dropna()
    trainX = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    trainY = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])

    test_path = os.path.join(data_path, "dataset_" + dataset_name + "-test.csv")
    dataframe = pd.DataFrame(
        pd.read_csv(test_path)).dropna()
    testX = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    testY = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    
    frames = [trainX, testX]
    X = pd.concat(frames)
    
    (n, m) = trainX.shape
    # print(m)
    
    if mode == "custom-bucket":
        num_buckets_per_feature = (max_splits // m) + 1
        encoder = Encoder(X.values[:,:], header=X.columns[:], 
                  mode="custom-bucketize", num_buckets_per_feature=num_buckets_per_feature)
        trainX = pd.DataFrame(encoder.encode(trainX.values[:,:]), columns=encoder.headers)
        testX = pd.DataFrame(encoder.encode(testX.values[:,:]), columns=encoder.headers)
    else:
        encoder = Encoder(X.values[:,:], header=X.columns[:])
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
    
    (n, z) = trainX.shape

    x_train, y_train = trainX.to_numpy().tolist(), trainY.to_numpy()[:, 0].tolist()
    x_test, y_test = testX.to_numpy().tolist(), testY.to_numpy()[:, 0].tolist()
    
    classes = sorted(list(set(y_train + y_test)))
    
    feature_value_dict = {}
    
    for i in range(len(x_train[0])): # iterate over all features
        vals = []
        for j in range(len(x_train)): # find all possible values of this feature
            vals.append(x_train[j][i])
        for j in range(len(x_test)):
            vals.append(x_test[j][i])
        distinct_vals = sorted(list(set(vals)))
        feature_value_dict[i] = distinct_vals
    
    num_features = len(feature_value_dict)
    
    class_string_to_int_mapper = {}
    cnt = 0
    for c in classes:
        class_string_to_int_mapper[c] = cnt
        cnt += 1
        
    for j in range(len(y_train)):
        y_train[j] = class_string_to_int_mapper[y_train[j]]
    for j in range(len(y_test)):
        y_test[j] = class_string_to_int_mapper[y_test[j]]
        
    classes = sorted(list(class_string_to_int_mapper.values()))
    
    return x_train, y_train, x_test, y_test, feature_value_dict, classes
