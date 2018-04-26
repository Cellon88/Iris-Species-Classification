# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:36:36 2018

@author: yhj
"""

import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split


# ###################################################################
# data loding
data = pd.read_csv('C:/data/Iris.csv')

# ###################################################################
# data check
data.head(10)
data.info()
data.isnull().sum()
data.describe()

species_list = data['Species'].unique()

# ###################################################################
# Feature Engineering
# Delet Id, Encoding 'Species'
data = data.drop(['Id'], axis=1)

data['Species'] = LabelEncoder().fit_transform(data['Species'].values)

# ###################################################################
# Exploratory Data Analysis (EDA)

# Iris Pairplot
sns.pairplot(data, hue='Species')

# sns.FacetGrid(data, hue='Species', size=7).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm').add_legend()
# sns.FacetGrid(data, hue='Species', size=7).map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm').add_legend()

sns.pairplot(data, x_vars=['SepalLengthCm', 'SepalWidthCm'], y_vars=['PetalLengthCm', 'PetalWidthCm'], hue='Species', size = 4)
sns.pairplot(data, x_vars=['SepalLengthCm', 'PetalLengthCm'], y_vars=['PetalWidthCm', 'SepalWidthCm'], hue='Species', size = 4)


# Species Distribution
fig = plt.figure(figsize=(12,7))
fig.suptitle('Species Distribution', fontsize=20)

ax1 = fig.add_subplot(221)
data.groupby(['Species']).PetalLengthCm.plot('hist', alpha=0.8, title='PetalLengthCm')
plt.legend(species_list, loc=1, fontsize='10')
ax2 = fig.add_subplot(222,sharey=ax1)
data.groupby(['Species']).PetalWidthCm.plot('hist', alpha=0.8, title='PetalWidthCm')
plt.legend(species_list, loc=1, fontsize='10')
ax3 = fig.add_subplot(223,sharey=ax1)
data.groupby(['Species']).SepalLengthCm.plot('hist', alpha=0.8, title='SepalLengthCm')
plt.legend(species_list, loc=1, fontsize='10')
ax4 = fig.add_subplot(224,sharey=ax1)
data.groupby(['Species']).SepalWidthCm.plot('hist', alpha=0.8, title='SepalWidthCm')
plt.legend(species_list, loc=1, fontsize='10')
plt.show()


# ###################################################################
# Explanatory variable X(설명변수), Response variable Y(반응변수)

X = data.drop(['Species'], axis=1)
y = data['Species']


# ###################################################################
# Data Split Ver.1

seed = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)


# ###################################################################
# Cross Vaildate - KFold Ver.1
# 데이터가 적기 때문에 5겹 Fold로 Cross Vaildate
# Regression에선 KFold가 기본, classifier에선 StratifiedKFold가 기본
# SEED = 100

# Kfold = KFold(n_splits=5, random_state=seed)
Kfold = StratifiedKFold(n_splits=5, random_state=seed)


# KFold             : 데이터 셋을 K개의 sub-set로 분리하는 방법. 
#                     분리된 K개의 sub-set 중 하나만 제외한 K-1개의 sub-sets를 training set으로 이용하여 K개의 모형 추정한다.
# ShuffleSplit      : 랜덤추출이기 때문에 한 샘플이 여러 폴드에 포함될수 있음
# RepeatedKFold     : 안정된 교차검증의 결과를 얻기 위해 지정한 횟수만큼 반복해서 무작위 분할
#                     반복할때 무작위성 때문에 분할기에 Shuffle=True 옵션이 자동으로 적용.  
# StratifiedKFold   : 계층별 데이터셋 분할. 타깃값 클래스 비율을 고려하여 분할함.
# RepeatedStratifiedKFold : 계층별 데이터셋 무작위 분할


# ###################################################################
# Visualization, Report Ver.2
# Ver.1 -> 2 : 컬러색상과 plot 크기변경

# GridSearchCV, RandomizedSearchCV Report Function 
# -> by. scikit-learn.org "Comparing randomized search and grid search for hyperparameter estimation"
def report(results, n_top=3):
    lcount = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            if lcount > 4:
                break
            lcount += 1
                
            

def model_scores(y_test,y_pred):
    acc = accuracy_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print('Accuracy : %.2f %%' % acc)
    print('F1 Score : %.2f ' % f1)
    print('Confusion Matrix :')
    print(conf_matrix)
    
    # sns.heatmap(conf_matrix, cmap='Greys', annot=True, linewidths=0.5)

    #print(classification_report(y_test, y_pred))
    # Warning : UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
    # when no data points are classified as positive, precision divides by zero as it is defined as TP / (TP + FP) (i.e., true positives / true and false positives). 
    # The library then sets precision to 0, but issues a warning as actually the value is undefined. 
    # F1 depends on precision and hence is not defined either.
    
    return {'acc':[acc], 'f1':[f1]}


def result_vis(acc_results_vis, f1_results_vis, names_vis):   
    fig =plt.figure(figsize=(5,5))
    fig.suptitle('Algorithm Comparison - Accuracy')
    ax = fig.add_subplot(111)
    plt.boxplot(acc_results_vis, vert=False)
    ax.set_yticklabels(names_vis)
    plt.show()

    fig =plt.figure(figsize=(5,5))
    fig.suptitle('Algorithm Comparison - F1')
    ax = fig.add_subplot(111)
    plt.boxplot(f1_results_vis, vert=False)
    ax.set_yticklabels(names_vis)
    plt.show()


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# ###################################################################
# Hyperparameter Grid Ver.1
param_grid_pipe = {
        'RandomForest' : {'RandomForest__n_estimators'      : st.randint(500, 1000),
                          'RandomForest__max_features'      : ['auto','sqrt','log2'],
                          'RandomForest__max_depth'         : st.randint(2, 100),
                          'RandomForest__min_samples_split' : st.randint(2, 100),
                          'RandomForest__min_samples_leaf'  : st.randint(2, 100),
                          'RandomForest__criterion'         : ['gini', 'entropy']},
                          
        'GBM'          : {'GBM__n_estimators'               : st.randint(500, 1000),
                          'GBM__max_depth'                  : st.randint(2, 100),
                          'GBM__learning_rate'              : st.uniform(0.001, 0.2),
                          'GBM__min_samples_split'          : st.randint(2, 100),
                          'GBM__min_samples_leaf'           : st.randint(2, 100)},
                          
        'XGB'          : {'XGB__n_estimators'               : st.randint(500, 1000),
                          'XGB__max_depth'                  : st.randint(2, 100),
                          'XGB__learning_rate'              : st.uniform(0.001, 0.2),
                          'XGB__colsample_bytree'           : st.beta(10, 1),
                          'XGB__subsample'                  : st.beta(10, 1),
                          'XGB__gamma'                      : st.uniform(0, 10),
                          'XGB__min_child_weight'           : st.expon(0, 10)}
}

cache_dir = mkdtemp()

# ###################################################################
# MODELS - Pipeline
models = []
models.append(('RandomForest _pipe', RandomizedSearchCV(Pipeline([('scaler', StandardScaler()), ('RandomForest', RandomForestClassifier())], memory=cache_dir), param_grid_pipe['RandomForest'], scoring='accuracy', cv=Kfold, n_jobs=1, n_iter=100)))
models.append(('GBM          _pipe', RandomizedSearchCV(Pipeline([('scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())], memory=cache_dir), param_grid_pipe['GBM'], scoring='accuracy', cv=Kfold, n_jobs=1, n_iter=100)))
models.append(('XGBoost      _pipe', RandomizedSearchCV(Pipeline([('scaler', StandardScaler()), ('XGB', xgb.XGBClassifier(booster='gbtree',objective='multi:softmax'))], memory=cache_dir), param_grid_pipe['XGB'], scoring='accuracy', cv=Kfold, n_jobs=1, n_iter=100)))


# ###################################################################
# MODELS Scores Ver.1
# 각 모델들을 차례대로 평가

acc_results =[]
f1_results =[]
names= []
for name, model in models:
    
    start = time()
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)
    
    print('')
    print('## %s ##################################' % name)
    print('Best  score : %.4f' % model.best_score_)
    print('Test  score : %.4f' % model.score(X_test, y_test))
    results = model_scores(y_test, y_pred)
    
    print("\n%s ParamsSearchCV took %.2f seconds for %d candidate parameter settings." 
          % (name.replace(" ", ""), time() - start, len(model.cv_results_['params'])))
    report(model.cv_results_)
    
    acc_results.append(results['acc'])
    f1_results.append(results['f1'])
    names.append(name)
    rmtree(cache_dir)
    


result_vis(acc_results, f1_results, names)


'''
RandomForest ParamsSearchCV took 740.82 seconds for 50 candidate parameter settings.
Model with rank: 1
Mean validation score: 0.421 (std: 0.106)
Parameters: {'RandomForest__criterion': 'gini', 'RandomForest__max_depth': 51, 'RandomForest__max_features': 'sqrt', 'RandomForest__min_samples_leaf': 12, 'RandomForest__min_samples_split': 17, 'RandomForest__n_estimators': 1083}

GBM ParamsSearchCV took 112.21 seconds for 100 candidate parameter settings.
Model with rank: 1
Mean validation score: 0.947 (std: 0.070)
Parameters: {'GBM__learning_rate': 0.08115438505005958, 'GBM__max_depth': 3, 'GBM__min_samples_leaf': 8, 'GBM__min_samples_split': 19, 'GBM__n_estimators': 488}

XGB ParamsSearchCV took 31.27 seconds for 100 candidate parameter settings.
Model with rank: 1
Mean validation score: 0.947 (std: 0.062)
Parameters: {'XGB__colsample_bytree': 0.9825553897248653, 'XGB__gamma': 7.477928348437604, 'XGB__learning_rate': 0.07981520928258301, 'XGB__max_depth': 82, 'XGB__min_child_weight': 2.772294984936991, 'XGB__n_estimators': 349, 'XGB__subsample': 0.8823208201017416}
'''


# ###################################################################
# Hyperparameter optimization by Skopt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# ###################################################################
# Hyperparameter Grid_Skopt Ver.1

# 숫자 범위 (시작, 끝)
# Categorical : 범주, prior(dafault=None), [list, shape=(categories,)], 각각의 카테고리별 선택 확률, 기본적으로 모두 같은 확률로 설정
# Real        : 실수, prior(default='uniform'), ['uniform','log-uniform'], log-uniform은 로그 스케일(log10)
# Integer     : 정수

param_grid_skopt = {
        'RandomForest' : {'RandomForest__n_estimators'      : Integer(1000, 10000),
                          'RandomForest__max_features'      : Categorical(['auto','sqrt','log2']),
                          'RandomForest__max_depth'         : Integer(2, 500),
                          'RandomForest__min_samples_split' : Integer(2, 500),
                          'RandomForest__min_samples_leaf'  : Integer(2, 500),
                          'RandomForest__criterion'         : Categorical(['gini', 'entropy'])},

        'GBM'          : {'GBM__n_estimators'               : Integer(1000, 10000),
                          'GBM__max_depth'                  : Integer(2, 500),
                          'GBM__learning_rate'              : Real(1e-6, 1e-1, 'log-uniform'),
                          'GBM__min_samples_split'          : Integer(2, 500),
                          'GBM__min_samples_leaf'           : Integer(2, 500)},
                          
        'XGB'          : {'XGB__booster'                    : Categorical(['gbtree','dart']),
                          'XGB__objective'                  : Categorical(['multi:softmax','multi:softprob']),
                          'XGB__n_estimators'               : Integer(1000, 10000),
                          'XGB__max_depth'                  : Integer(2, 500),
                          'XGB__learning_rate'              : Real(1e-6, 1e-1, 'log-uniform'),
                          'XGB__subsample'                  : Real(0, 1, 'uniform'),
                          'XGB__gamma'                      : Integer(0, 100),
                          'XGB__min_child_weight'           : Real(1e-2, 1e+2, 'log-uniform')}
}

cache_dir = mkdtemp()

# ###################################################################
# MODELS - Skopt (BayesSearchCV)
# Bayesian optimization over hyper parameters
# Gaussian Process는 함수의 확률적 분포를 modeling 할 수 있게 해준다.
# x를 y에 매핑하는 함수를 가정하고, 그 함수를 추론. 베이지한 추론은 데이터로부터 함수 자체에 대한 추론을 수행한다.
# Gaussian Process로 에러 함수에 대한 근사함수를 만든 다음 그 근사함수 상에서 에러를 최소화하는, 또는 근사 함수를 좀더 정확하게 만들 수 있는 점들을 다음 실험의 하이퍼파라미터로 사용한다.
# Gaussian Process는 관촬되지 않은 매개변수에 대한 가정을 하기위해 이전에 평가 된 매개변수 세트와 결과정확도를 사용
# ###################################################################
models = []
models.append(('RandomForest', BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('RandomForest', RandomForestClassifier())], memory=cache_dir), param_grid_skopt['RandomForest'], n_iter = 50, n_jobs=1, cv=Kfold)))
models.append(('GBM         ', BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())], memory=cache_dir), param_grid_skopt['GBM'], n_iter = 50, n_jobs=1, cv=Kfold)))
models.append(('XGBoost     ', BayesSearchCV(Pipeline([('scaler', StandardScaler()), ('XGB', xgb.XGBClassifier())], memory=cache_dir), param_grid_skopt['XGB'], n_iter = 50, n_jobs=1, cv=Kfold)))
# ###################################################################
# MODELS Scores Ver.1
# 각 모델들을 차례대로 평가

acc_results =[]
f1_results =[]
names= []
for name, model in models:
    
    start = time()
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)
    
    print('## %s ##################################' % name)
    print('Best  score : %.4f' % model.best_score_)
    print('Test  score : %.4f' % model.score(X_test, y_test))
    results = model_scores(y_test, y_pred)
    
    print("\n%s BayesSearchCV took %.2f seconds for %d candidate parameter settings." 
          % (name.replace(" ", ""), time() - start, len(model.cv_results_['params'])))
    print("Best Parameters : ", model.best_params_)
    #report(model.cv_results_)
    
    acc_results.append(results['acc'])
    f1_results.append(results['f1'])
    names.append(name)
    rmtree(cache_dir)
    

result_vis(acc_results, f1_results, names)


'''
RandomForest BayesSearchCV took 1391.30 seconds for 50 candidate parameter settings.
Best Parameters :  {'RandomForest__criterion': 'gini', 'RandomForest__max_depth': 500, 'RandomForest__max_features': 'sqrt', 'RandomForest__min_samples_leaf': 2, 'RandomForest__min_samples_split': 2, 'RandomForest__n_estimators': 1000}

GBM BayesSearchCV took 1265.90 seconds for 50 candidate parameter settings.
Best Parameters :  {'GBM__learning_rate': 0.0013925775747943296, 'GBM__max_depth': 494, 'GBM__min_samples_leaf': 2, 'GBM__min_samples_split': 2, 'GBM__n_estimators': 10000}




'''
