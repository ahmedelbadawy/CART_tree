import pandas as pd 
import numpy as np
import itertools
import ast

def  gini_impurity(y):
    p1 = len(y[y == 1])/len(y)
    p0 = len(y[y == 0])/len(y)
    return (1 - p1 ** 2 - p0 ** 2)
def entropy(y):
    # y = list(y)
    p1 = len(y[y == 1])/len(y)
    p0 = len(y[y == 0])/len(y)
    return(-p1 * np.log2(p1 + 1e-9) - p0 * np.log2(p0 + 1e-9))

def information_gain(y, cut,func):
    if len(cut[cut==1]) == len(cut) or len(cut[cut==0]) == len(cut):
        ig = 0
    else:
        ig = func(y) - ((len(y[cut])/len(y)) * func(y[cut]) + (len(y[~(cut)])/len(y)) * func(y[~(cut)]))
    return ig

def categorical_options(x):

  x = x.unique()

  optiones = []
  for L in range(len(x)+1):
      for subset in itertools.combinations(x, L):
          optiones.append(list(subset))
  return optiones[1:-1]

def best_split(x,y,func):
    ig_list = []
    cut_list = []

    if len(x.unique()) < 10:
        options = categorical_options(x)
        for val in options:
            cut = x.isin(val)
            ig = information_gain(y,cut,func)
            ig_list.append(ig)
            cut_list.append(val)
    else:
        options = x.sort_values().unique()[1:] # for not taking the first value
        for val in options:
            cut = x < val
            ig = information_gain(y,cut,func)
            ig_list.append(ig)
            cut_list.append(val)
    if len(ig_list) == 0:
        max_ig =0
        best_cut = None
    else:
        max_ig = max(ig_list)
        best_cut = cut_list[ig_list.index(max_ig)]
    return max_ig, best_cut
def data_best_split(data,func):
    if len(data)<2:
        best_ig = 0
        best_cut_value = 0
        best_variable = 0
    else:
        ig_list = []
        cut_list = []
        for i in data.columns.values[:-1]:
            
            ig,cut_val = best_split(data[i],data.iloc[:,-1],entropy)
            ig_list.append(ig)
            cut_list.append(cut_val)

        best_ig = max(ig_list) 
        best_cut_value = cut_list[ig_list.index(best_ig)]
        best_variable = data.columns.values[ig_list.index(best_ig)]

    return best_ig , best_cut_value , best_variable

def split(feature,x,cut_value):
    if len(x[feature].unique()) < 10:
        left_data = x[x[feature].isin(cut_value)]
        right_data = x[~x[feature].isin(cut_value)]
        categorical_feature = True
    else:
        left_data = x[ x[feature]< cut_value]
        right_data = x[ x[feature] >= cut_value]
        categorical_feature = False
    return left_data, right_data , categorical_feature


def train_data(data,func,max_depth = 5, depth = 0):
    # counter += 1
    if depth < max_depth:
        ig, cut_value, cut_variable = data_best_split(data,func)
        if ig > 1e-9:
            right_data , left_data , categorical_feature = split(cut_variable,data,cut_value)
            depth += 1
            if categorical_feature:
                split_condition = "in"
            else:
                split_condition = "<"
            condition =   "{}${}${}".format(cut_variable,split_condition,cut_value)
            subtree = {condition: []}
            split_lift = train_data(left_data,func ,max_depth,depth)
            split_right = train_data(right_data,func ,max_depth, depth)
            if split_lift == split_right:
                subtree = split_lift

            else:
                subtree[condition].append(split_lift)
                subtree[condition].append(split_right)
        else:
            predicted_value = round(data.iloc[:,-1].mean())
            return predicted_value
    else: 
        predicted_value = round(data.iloc[:,-1].mean())
        return predicted_value
    return subtree

def predict_singel_value(x,tree):
    condition = list(tree.keys())[0]
    if condition.split("$")[1] == "in":
        if x[condition.split("$")[0]] in ast.literal_eval(condition.split("$")[2]):
            answer = tree[condition][0]
        else:
            answer = tree[condition][1]
    else:
        if x[condition.split("$")[0]] < float(condition.split("$")[2]):
            answer = tree[condition][0]
        else:
            answer = tree[condition][1]
    if type(answer) is dict:
        return predict_singel_value(x,answer)
    else:
        return answer

def predict(data,tree):
    values = []
    for i in range(len(data)):
        
        values.append(predict_singel_value(data.iloc[i,:],tree))
        
    return values


def get_accuracy(predicted_values,y):
    return list(predicted_values == y).count(1)/len(predicted_values)
    

