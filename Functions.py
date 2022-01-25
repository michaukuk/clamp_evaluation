# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:37:29 2021

@author: Edyta
"""
import pandas as pd
import shap
import pickle
import itertools
import math
import statistics
import re
import lime
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
from functools import partial
from multiprocessing import Pool, cpu_count
import collections
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import RandomizedSearchCV




def load_dataframe(name):
    with open ('datasets/'+str(name)+'.pickle', 'rb') as f:
        df_raw = pickle.load(f)
        #df_raw = df_raw[0]
    #try:
    #    df = convert_df(data = df_raw.drop('MATID', axis = 1), resample = False, res_ind = 1)
    #except: 
        df = convert_df(df_raw)
    return df

def convert_df(data, resample = False, res_ind = 2):
    #temp_cols = data['y']
    #data.drop(columns = 'y', inplace = True)
    #data['y'] = temp_cols
    if resample:
        data = data.iloc[::res_ind, :]
    data = data.reset_index(drop='index')
    return data

def prepare_df(data, test_size = 0.2):
    df_features = data.columns.drop(data.columns[-2:])
    num_of_df_features = len(df_features)
    class_names = data[data.columns[-1]].unique()
    X_train, X_test, y_train, y_test = train_test_split(data.drop(data.columns[-1], axis = 1), data[data.columns[-1]], test_size=test_size, random_state=0)
    return X_train, X_test, y_train.values, y_test.values, class_names, num_of_df_features, df_features

def optmize_booster(X_train, X_test, y_train, y_test, average, param_grid = {'eta': [0.001, 0.01, 0.2],
                'max_depth': [10, 30, 50],
                'alpha': [0.1, 0.4],
                #'scale_pos_weight': [0.14],
                 'max_delta_step': [0.5, 0.7],
                 'tree_method': ['exact']}):
    
    best_score = -1e-6
    best_params = {}
    best_model = None
    
    unique, counts = np.unique(y_test, return_counts=True)
    d = dict(zip(unique, counts))
    bad_label = min(d, key=d.get)
    
    
    # iterate through grid to find the best model
    it_number = 0
    compute = 1
    for x in param_grid.values():
        compute = compute*len(x)
    print('Total iterations: ', compute)
          
    #---------------------------------------------------------------
    clf = RandomizedSearchCV(xgboost.XGBClassifier(), param_grid, n_iter = compute, n_jobs = -1)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    '''
    for values in itertools.product(*param_grid.values()):
        point = dict(zip(param_grid.keys(), values))
        settings = {**point}
        model = xgboost.XGBClassifier(**settings)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    '''
    if len(np.unique(y_test)) == 2:
        score = f1_score(y_test, y_pred, average="binary", pos_label=bad_label)
        recall = recall_score(y_test, y_pred, average="binary", pos_label=bad_label)
        precision = precision_score(y_test, y_pred, average="binary", pos_label=bad_label)
            
    else:
        score = f1_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        precision = precision_score(y_test, y_pred, average=average)
        
    accuracy = accuracy_score(y_test, y_pred)
    '''
        if score > best_score:
            best_score = score
            recall = recall
            precision = precision
            accuracy = accuracy
            best_params = settings
            best_model = model
        
        print('Recall: ', round(recall,2), 
              'f1: ', round(best_score,2),
              'precision: ', round(precision,2),
              'accuracy: ', round(accuracy,2),
              'average: ', average)
    '''
    best_score = score
    recall = recall
    precision = precision
    accuracy = accuracy
    best_model = model
    best_params = model.best_params_
    
    print('Recall: ', round(recall,2), 
              'f1: ', round(best_score,2),
              'precision: ', round(precision,2),
              'accuracy: ', round(accuracy,2),
              'average: ', average)
    
    # print out results
    print("Best parameters:", best_params)
    print("Best score:", round(best_score,3))
    return best_model, score, recall, precision, accuracy, best_params


def randomly_selected(data, class_names, procent_of_points):
    temp_list = []
    number_of_points = math.ceil(data.shape[0]*procent_of_points/100)
    X = data.drop('y', axis = 1).to_numpy()
    y = data['y'].to_numpy()
    for cluster in class_names:
        #number_of_points = math.ceil(data[data['y'] == cluster].shape[0]*procent_of_points/100)
        X_t = data[data['y'] == cluster]
        print((X_t.sample(n = number_of_points)).shape)
        temp_list.append(X_t.sample(n = number_of_points))
    df_ready = pd.concat(temp_list)
    df_ready = df_ready.set_index('y')
    return df_ready

def tree_query(data, class_names, procent_of_points, metrics = 'minkowski'):
    rng = np.random.RandomState(0)
    number_of_points = math.ceil(data.shape[0]*procent_of_points/100)
    temp_list = []
    X = data.drop('y', axis = 1).to_numpy()
    y = data['y'].to_numpy()
    print("y", y)
    clf = NearestCentroid()
    clf.fit(X, y)
    df_centroids = pd.DataFrame(clf.centroids_, columns = data.columns[:-1])
    for cluster in class_names:
       # number_of_points = math.ceil(data[data['y'] == cluster].shape[0]*procent_of_points/100)
        X_t = data[data['y'] == cluster].drop('y', axis = 1).values  # 10 points in 3 dimensions
        tree = KDTree(X_t, leaf_size = 10, metric = metrics)  
        dist, ind = tree.query(df_centroids.iloc[cluster].values.reshape(1,-1), k=len(X_t))      
        temp_df = data[data['y'] == cluster].iloc[ind[0][-number_of_points:],:]
        print(temp_df.shape)
        temp_list.append(temp_df)

    df_ready = pd.concat(temp_list)
    df_ready = df_ready.set_index('y')
    
    return df_ready

def outliers(data, class_names, cont):
    temp_list = []
    number_of_points = math.ceil(data.shape[0]*cont)
    for cluster in class_names:
        cluster_cont = number_of_points/(data[data['y'] == cluster].shape[0])
        random_data = data[data['y'] == cluster].drop('y', axis = 1).values
        print("random_data", random_data)
        clf = IsolationForest(random_state = 1, contamination= cluster_cont)
        preds = clf.fit_predict(random_data)
        df_temp = pd.DataFrame(random_data[[i for i, x in enumerate(preds) if x == -1]], columns = data.columns[:-1])
        df_temp['y'] = [cluster] * len(df_temp)
        print(df_temp.shape)
        temp_list.append(df_temp)
    df_ready = pd.concat(temp_list)
    df_ready = df_ready.set_index('y')
    print(df_ready)
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

def anchor_exp(data, average = 'macro', test_size = 0.2, cont = 0.1, procent_of_points = 10, description_method = 'tree_query', threshold = 0.85, metrics = 'minkowski', model = None):
    
    original_data_with_matid = data.copy(deep = True)
    X_train_with_matid, X_test_with_matid, y_train, y_test, class_names, num_of_df_features, df_features = prepare_df(original_data_with_matid, test_size)
    X_train = X_train_with_matid.drop('MATID', axis = 1).values
    X_test = X_test_with_matid.drop('MATID', axis = 1).values

    data = X_train_with_matid.drop('MATID', axis = 1)
    data['y'] = y_train
    #data = original_data_with_matid.drop('MATID', axis = 1)
    orignal_features = list(data.columns[:-1])
    data.columns = ['F' + str(list(data.columns).index(x)+1) for x in list(data.columns)]
    data.columns = [*data.columns[:-1], 'y']
    modyfied_features = list(data.columns[:-1])
    df_features = modyfied_features
        
    #coef = sum(data['y'] ==1)/sum(data['y'] ==0)   
    
    if model is None:
        model, score, recall, precision, accuracy, best_params = optmize_booster(X_train, X_test, y_train, y_test, average)
        with open('scores.pickle', 'wb') as f:
            pickle.dump([score, recall, precision, accuracy],f)
    model.fit(X_train, y_train)
    predict_fn = lambda x: model.predict_proba(x)
    explainer = AnchorTabular(predict_fn, df_features)
    explainer.fit(X_train, disc_perc=[25, 50, 75])

    list_of_rules = []
    if description_method == 'tree_query':
        func = tree_query
        additional_input = [data, class_names, procent_of_points, metrics]
    elif description_method == 'outlier':
        func = outliers
        additional_input = [data, class_names, cont]
    elif description_method == 'random':
        func = randomly_selected
        additional_input = [data, class_names, procent_of_points]
        
    print("Test size:", str(collections.Counter(y_test)).replace('Counter',""))
    print(data)
    print(class_names)
    print(cont)
    df_input = func(*additional_input)
    #print('Number of description points: ', df_input.shape[0])
    print(df_input)
    rules = []
    class_names = np.unique(df_input.index)
    rules_out = pd.DataFrame()
    for cluster in class_names:
        anchors = []
        for idx in range(len(df_input.values)):
            #indexes = explainer.predictor(df_input.values[idx].reshape(1, -1))[0]
            #if class_names[indexes] == cluster:
             if df_input.index[idx] == cluster:   
                explanation = explainer.explain(df_input.values[idx], threshold=threshold)
                exp = explanation.anchor
                anchors.append('Anchor: %s' % (' AND '.join(exp)))
                    #if scores:
                    #    print("-------------")
                    #    print('Cluster no: ', cluster)
                    #    print('Precision: %.2f' % explanation['precision'])
                    #    print('Coverage: %.2f' % explanation['coverage'])
                    #    print('Rule: ', anchors)
                rules_out = rules_out.append({'Rule': (' AND '.join(exp)), 'Precision': explanation['precision'], 'Coverage': explanation['coverage'], 'Cluster': cluster},                                                        ignore_index = True) 
    print(rules_out.shape)
    rules_out['Rule'].replace('', np.nan, inplace=True)
    rules_out.dropna(subset=['Rule'], inplace=True)
    rules_out.drop_duplicates(inplace = True, ignore_index = True)
    rules_out_output = rules_out.copy(deep = True)
    rules_out_output = convert_best_rules(rules_out_output)
    
    df2hmr(rules_out_output, 'model_anchor_' + func.__name__ + '.hmr', df_columns_names = list(df_features),  numfeats = num_of_df_features)
    
    rules_out_output = convert_rules(rules_out_output, orignal_features, modyfied_features)
    
    rules_out_output.to_csv('scores/Rules_out_' + func.__name__ + '.csv', sep='\t', index = False)
    print(f'Number of created rules: {len(rules_out_output)}')
        
    return rules_out_output, X_test, y_test, model, len(rules_out_output), X_train_with_matid, X_test_with_matid, df_input.shape[0], best_params



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



def lime_exp(data, test_size = 0.8, cont = 0.1, max_samp = 50, number_of_points = 10, description_method = 'tree_query', metrics = 'minkowski'):
    
    orignal_features = list(data.columns[:-1])
    data.columns = ['F' + str(list(data.columns).index(x)+1) for x in list(data.columns)]
    data.columns = [*data.columns[:-1], 'y']
    modyfied_features = list(data.columns[:-1])
    
    X_train, X_test, y_train, y_test, class_names, num_of_df_features, df_features = prepare_df(data)
    
    #model, score = optmize_booster(X_train, X_test, y_train, y_test)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)
    lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric':'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': None,
    'num_iteration':100,
    'num_threads':7,
    'max_depth':12,
    'min_data_in_leaf':100,
    'alpha':0.5}

    model = lgb.train(lgb_params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)
    
    def prob(data):
        return np.array(list(zip(1-model.predict(data),model.predict(data))))

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=df_features, class_names=class_names, verbose=True, mode='classification')
    if num_of_df_features > 5:
        num_features_to_explain = 5
    else:
        num_features_to_explain = num_of_df_features

    print('Multi dimensions')
    print("Test size:", str(collections.Counter(y_test)).replace('Counter',""))
    
    if description_method == 'tree_query':
        func = tree_query
        additional_input = [data, class_names, number_of_points, metrics]
    elif description_method == 'outlier':
        func = outliers
        additional_input = [data, class_names, cont, max_samp]

    df_input = func(*additional_input)
        
    rule_list_lime = []
    cert_list = []
    cluster_label = []
    lime_rules = pd.DataFrame()
    for index in range(len(df_input.values)):
        exp = explainer.explain_instance(df_input.values[index], prob, num_features=num_features_to_explain)
        rule_list_lime.append(" AND ".join([i[0] for i in exp.as_list()]))
        cert_list.append(abs(sum([i[1] for i in exp.as_list()])))
        cluster_label.append(df_input.index[index])
            
   
    lime_rules = pd.DataFrame({'Cluster':cluster_label, 'Coverage': 1, 'Precision': cert_list, 'Rule':rule_list_lime})
    
    lime_rules.drop_duplicates(inplace = True, ignore_index = True)
    
    df2hmr(lime_rules, 'model_lime_' + func.__name__ + '.hmr', df_columns_names = list(df_features),  numfeats = num_of_df_features)
    
    lime_rules = convert_rules(lime_rules, orignal_features, modyfied_features)
    
    print(f'Number of created rules: {len(lime_rules)}')
    return lime_rules, X_test, y_test
    
def plot_conff(df, label):
    Conf_df = df.copy(deep = True) 
    Conf_df['heartdroid_output'] = label
    Conf_df.rename(columns = {'y': 'orignal_lables'}, inplace=True)
    M = Conf_df.pivot_table(index='heartdroid_output',columns='orignal_lables',values='F1',aggfunc='count').fillna(0)
    g,axs = plt.subplots(figsize=(15,7))
    axs = sns.heatmap(M.astype(int), annot=True, cmap='viridis',fmt='g')
    return None


def parallel_task(dataset, task_wrapper, threads = None):
    if threads is None:
        threads = cpu_count()
    with Pool(threads) as p:
        #print(1)
        #dataset_split = np.array_split(dataset, cpu_count())
        #ret_list = []
        #print(2)
       # 
       # for thresh in dataset_split:
       #     ret_list.append(p.task_wrapper(thresh))
       ## 
        #print(3)
        ret_list = p.map(task_wrapper, np.array_split(dataset, threads))
        p.close()
    scores = pd.concat([pd.DataFrame(ret_list[x][0]) for x in range(len(ret_list))], sort=True)
    labels = pd.concat([pd.DataFrame(ret_list[x][1]) for x in range(len(ret_list))], sort=True)
    return scores, labels

