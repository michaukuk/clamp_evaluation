# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:37:29 2021

@author: Edyta
"""
import pandas as pd
import pickle
import itertools
import math
import statistics
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, auc
from sklearn.preprocessing import LabelEncoder
from alibi.explainers import AnchorTabular
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import RandomizedSearchCV
import os
from IPython.display import Image, display
from sklearn_extra.cluster import KMedoids


def prepare_df(data, test_size):
    df_features = data.columns[:-1]
    num_of_df_features = len(df_features)
    class_names = data[data.columns[-1]].unique()
    X_train, X_test, y_train, y_test = train_test_split(data.drop(data.columns[-1], axis = 1), data[data.columns[-1]], test_size=test_size, random_state=0)
    return X_train, X_test, y_train.values, y_test.values, class_names, num_of_df_features, df_features

def optmize_booster(X_train, X_test, y_train, y_test, average, param_grid = {'eta': [0.001, 0.01, 0.2],
                'max_depth': [10, 30, 50],
                'alpha': [0.1, 0.4],
                'max_delta_step': [0.5, 0.7],
                'tree_method': ['exact']}):

    it_number = 0
    compute = 1
    for x in param_grid.values():
        compute = compute*len(x)
    print('Total iterations: ', compute)
          
    clf = RandomizedSearchCV(xgboost.XGBClassifier(), param_grid, n_iter = compute)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = f1_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    accuracy = accuracy_score(y_test, y_pred)
    
    print('Recall: ', round(recall,2), 
              'f1: ', round(score,2),
              'precision: ', round(precision,2),
              'accuracy: ', round(accuracy,2),
              'average: ', average)
    
    return model, score, recall, precision, accuracy, model.best_params_


def randomly_selected(data, class_names, number_of_points):
    
    if number_of_points > data.shape[0]:
      number_of_points = data.shape[0]
      print(f'Number of points to training was limited to the maximum size of dataframe: {number_of_points}')
    X = data.drop('y', axis = 1).to_numpy()
    y = data['y'].to_numpy()

    temp_list = []
    for cluster in class_names:
        X_t = data[data['y'] == cluster]
        temp_list.append(X_t.sample(n = number_of_points))
    df_ready = pd.concat(temp_list)
    df_ready = df_ready.set_index('y')
    return df_ready

def tree_query(data, class_names, number_of_points, metrics):
    
    if number_of_points > data.shape[0]:
      number_of_points = data.shape[0]
      print(f'Number of points to training was limited to the maximum size of dataframe: {number_of_points}')

    X = data.drop('y', axis = 1).to_numpy()
    y = data['y'].to_numpy()

    clf = NearestCentroid()
    clf.fit(X, y)
    df_centroids = pd.DataFrame(clf.centroids_, columns = data.columns[:-1])
    
    temp_list = []
    for cluster in class_names:
        X_t = data[data['y'] == cluster].drop('y', axis = 1).values
        tree = KDTree(X_t, leaf_size = 10, metric = metrics)  
        dist, ind = tree.query(df_centroids.iloc[cluster].values.reshape(1,-1), k=len(X_t))      
        temp_df = data[data['y'] == cluster].iloc[ind[0][-number_of_points:],:]
        temp_list.append(temp_df)
    df_ready = pd.concat(temp_list)
    df_ready = df_ready.set_index('y')
    return df_ready

def outliers(data, class_names, cont):
    temp_list = []
    for cluster in class_names:
        random_data = data[data['y'] == cluster].drop('y', axis = 1).values
        clf = IsolationForest(random_state = 1, contamination= cont)
        preds = clf.fit_predict(random_data)
        df_temp = pd.DataFrame(random_data[[i for i, x in enumerate(preds) if x == -1]], columns = data.columns[:-1])
        df_temp['y'] = [cluster] * len(df_temp)
        temp_list.append(df_temp)
    df_ready = pd.concat(temp_list)
    df_ready = df_ready.set_index('y')
    return df_ready

def kmedoids_des(df, class_names):
    X = df.drop('y', axis = 1).to_numpy()
    kmedoids = KMedoids(n_clusters=len(df['y'].unique()), random_state=0).fit(X)
    df_ready = pd.DataFrame(data = kmedoids.cluster_centers_, columns = df.drop('y', axis = 1).columns)
    df_ready['y'] = df['y'].unique()
    df_ready = df_ready.set_index('y')
    return df_ready

def inparse(condition):
    fs = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\2',condition)
    ss =re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\4',condition)
    res=None
    if fs == '<':
        val = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\1',condition)
        res = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\3 in ['+str(eval(val)+0.001)+r' to \5 ]',condition)
    if ss == '<':
        val = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\5',condition)
        res = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\3 in [\1 to '+str(eval(val)-0.001)+']',condition)
    if res is None:
        return condition
    else:
        return res

def tohmr(series):
    result =[]
    for v in  series.split('AND'):
        v = inparse(v.strip().lower().replace(' ',''))
        result.append(v.replace('<=',' lte ')
                      .replace('>=',' gte ').replace('<',' lt ').replace('>',' gt ').replace('=','eq').lower())
    return '['+','.join(result)+']'

types = """xtype [name: float,
    domain: [-10000 to 10000],
    scale: 0,
    base: numeric
    ].

xtype [name: clustertype,
    domain: [0 to 1000],
    scale: 0,
    base: numeric
    ]."""

atts_cluster = """xattr [name: cluster,
    type: clustertype,
    class: simple,
    comm: out
    ].
"""

atts_placeholder = """
xattr [name: __NAME__,
    type: float,
    class: simple,
    comm: out
    ].
"""

schema_placeholder = """xschm anchor: [__NAME__] ==> [cluster].
"""

def df2hmr(data, filename, df_columns_names, confidence='product', numfeats = 2):
    data['hmr_cond'] = data['Rule'].apply(tohmr)  
    data['confidence'] = data['Coverage']*data['Precision']
    #numfeats = max([int(elem[-1]) for elem in data['Rule'][1].split(' ') if "F" in elem])
    atts = ''
    schemacond = []
    for i in range(1,numfeats+1):
        atts+=atts_placeholder.replace('__NAME__','f'+str(i))
        schemacond.append('f'+str(i))
    #for i in range(1,numfeats+1):
     #   atts+=atts_placeholder.replace('__NAME__', df_columns_names[i-1])
      #  schemacond.append(df_columns_names[i-1])

    schema = schema_placeholder.replace('__NAME__',','.join(schemacond))

    with open(filename,'w') as f:
        #print(types)
        #print(atts)
        #print(atts_cluster)
        #print(schema)
        f.write(types)
        f.write(atts) 
        f.write(atts_cluster) 
        f.write(schema) 
        for i,r in data.iterrows():
            
            
            #print('xrule anchor/'+str(i)+': '+r['hmr_cond']+ ' ==>  [cluster set '+str(r['Cluster'])+']. #'+str(r['confidence']))
 
            f.write('xrule anchor/'+str(i)+': '+r['hmr_cond']+ ' ==>  [cluster set '+str(r['Cluster'])+']. #'+str(r['confidence'])+'\n')

def anchor_exp(data_c, 
              average = 'weighted', 
              test_size = 0.3, 
              cont = 0.1, 
              number_of_points = 10, 
              description_method = 'tree_query', 
              threshold = 0.85, 
              metrics = 'minkowski'):
    
    
    data = data_c.copy(deep = True)
    
    X_train, X_test, y_train, y_test, class_names, num_of_df_features, df_features = prepare_df(data, test_size)

    orignal_features = list(data.columns[:-1])
    data.columns = ['F' + str(list(data.columns).index(x)+1) for x in list(data.columns)]
    data.columns = [*data.columns[:-1], 'y']
    modyfied_features = list(data.columns[:-1])
    df_features = modyfied_features
    
    model, score, recall, precision, accuracy, best_params = optmize_booster(X_train.values, X_test.values, y_train, y_test, average)
    predict_fn = lambda x: model.predict_proba(x)
    explainer = AnchorTabular(predict_fn, df_features)
    explainer.fit(X_train.values, disc_perc=[25, 50, 75])

    list_of_rules = []
    if description_method == 'tree_query':
        func = tree_query
        additional_input = [data, class_names, number_of_points, metrics]
    elif description_method == 'outlier':
        func = outliers
        additional_input = [data, class_names, cont]
    elif description_method == 'random':
        func = randomly_selected
        additional_input = [data, class_names, number_of_points]
    elif description_method == 'centroids':
        func = kmedoids_des
        additional_input = [data, class_names]
        
    df_input = func(*additional_input)
    
    rules = []
    rules_out = pd.DataFrame()
    for cluster in class_names:
        anchors = []
        for idx in range(len(df_input.values)):
             if df_input.index[idx] == cluster:   
                explanation = explainer.explain(df_input.values[idx], threshold=threshold)
                exp = explanation.anchor
                anchors.append('Anchor: %s' % (' AND '.join(exp)))
                rules_out = rules_out.append({'Rule': (' AND '.join(exp)), 'Precision': explanation['precision'], 'Coverage': explanation['coverage'], 'Cluster': cluster},                                                        ignore_index = True) 
    rules_out['Rule'].replace('', np.nan, inplace=True)
    rules_out.dropna(subset=['Rule'], inplace=True)
    rules_out.drop_duplicates(inplace = True, ignore_index = True)
    rules_out_output = rules_out.copy(deep = True)
    rules_out_output = convert_best_rules(rules_out_output)
    
    df2hmr(rules_out_output, 'model_anchor_' + func.__name__ + '.hmr', df_columns_names = list(df_features),  numfeats = num_of_df_features)
    
    rules_out_output = convert_rules(rules_out_output, orignal_features, modyfied_features)
    
    rules_out_output.to_csv('Rules_out_' + func.__name__ + '.csv', sep='\t', index = False)
    print(f'Number of created rules: {len(rules_out_output)}')
        
    return rules_out_output, X_test, y_test, model, len(rules_out_output), df_input.shape[0], best_params

def convert_best_rules(rules):
    rules['Cov_Pre'] = rules['Coverage']*rules['Precision']
    unique_rules = np.unique(rules['Rule'])
    temp_finall_rule_list = []
    for idx in unique_rules:
        temp_finall_rule_list.append(rules[rules['Rule'] == idx].sort_values(by = ['Cov_Pre'], ascending = False).head(1))
    rules_finall = pd.concat(temp_finall_rule_list)
    rules_finall.sort_values(by = ['Cluster'], inplace = True)
    rules_finall.reset_index(inplace = True)
    rules_finall.drop(['index', 'Cov_Pre'], axis = 1,  inplace = True)
    return rules_finall

def convert_rules (rules, org, mod):
    my_dict = {} 
    for key in mod: 
        for value in org: 
            my_dict[key] = value 
            org.remove(value) 
            break 
    rules_con_org_features = []
    for rule in rules['Rule']:
        temp = rule.split()
        res = []
        for wrd in temp:
            res.append(my_dict.get(wrd, wrd))
        rules_con_org_features.append(' '.join(res))
    rules['Rule'] = rules_con_org_features
    return rules


def load_data_to_explain(name, plot = True):
    df = pd.read_csv('datasets/'+name)
    
    if plot:
        i = Image(filename='pix/'+name.split('.')[0]+'.png')
        display(i)
    return df

def available_data():
    return os.listdir('datasets')
