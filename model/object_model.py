import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from category_encoders import WOEEncoder, SumEncoder
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from numpy.random import RandomState
from skopt import BayesSearchCV
from ..plot import plot_core as plot

class BaseModel(object):
    """
    A single API for various models
    """
    def __init__(self, X, y, auto_prep: bool = True):
        self.X = X
        self.y = y
        self.auto_prep = auto_prep
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 100)

    def __get_info_about_split(self, X_train, X_test, y_train, y_test, plot=False):
        tmp_df = pd.DataFrame({'Split name':['X_train', 'X_test', 'y_train', 'y_test'],
        'Value':[X_train, X_test, y_train, y_test]})
        if plot:
            plot.bar_plot(tmp_df['Split name'], tmp_df['Value'])
        else:
            print('X_train -', X_train.shape[0], '\nX_test -', X_test.shape[0], '\ny_train -', y_train.shape[0], '\ny_test -', y_test.shape[0])

    if auto_prep:
        cat_features = X_train.dtypes[X_train.dtypes == 'object'].index
        num_features = X_train.dtypes[X_train.dtypes != 'object'].index
        kf = KFold(n_splits=5, shuffle=True, random_state=123)

        def 

