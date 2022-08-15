# import os
# from numpy import genfromtxt
import numpy as np
# import matplotlib.pyplot as plt
# import sklearn
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import tensorflow_probability as tfp
# from sklearn.model_selection import train_test_split
# import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.ensemble import ExtraTreesClassifier # ET
# from sklearn.ensemble import AdaBoostClassifier # ADB
# from sklearn.svm import SVC # svc
from sklearn.neural_network import MLPClassifier # MLP
# from xgboost import XGBClassifier # XGB
# from sklearn.gaussian_process import GaussianProcessClassifier # GPC
# from sklearn.naive_bayes import GaussianNB # GNB
from sklearn.linear_model import LogisticRegression # LR
# from sklearn.ensemble import GradientBoostingClassifier # GBC
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# import scipy
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# import statistics
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Input
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras.layers import Dropout, BatchNormalization
np.set_printoptions(precision=3, suppress=True)

tfd = tfp.distributions

class moe:
    def __init__(self, parameters, HP, dataset_name):
        self.datapath = parameters[dataset_name]
        self.resname= dataset_name
        self.test_size = parameters['test_size']
        self.label = parameters['label']
        self.random_state = parameters['random_state']
        self.models_list = parameters['models_list']
        self.model_RF = RandomForestClassifier(n_estimators=10,
                                               max_depth=None,
                                               min_samples_split=2,
                                               random_state=0)

        self.model_ET = ExtraTreesClassifier(n_estimators=100,
                                             criterion='gini',
                                             min_samples_split=2)

        self.model_MLP = MLPClassifier(hidden_layer_sizes=(100,),
                                       activation='relu',
                                       solver='adam',
                                       alpha=0.0001)

        self.model_LR = LogisticRegression(solver='newton-cg',
                                           C=100)
        self.Data_Prepration()
        #self.results_dataframe = pd.DataFrame(columns=['method', 'Accuracy', 'sensitivity', 'specificity', 'f1'])
        self.experience_path = HP
        self.moe_Inputs= {}

    def Data_Prepration(self):
        df_input= pd.read_excel(self.datapath)
        Y = df_input.pop(self.label)
        X = df_input
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size,
                                                            random_state=self.random_state)
        self.cv = KFold(n_splits=10, random_state=1, shuffle=True)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print("Data Preparation Done!")

    def models_Prepration(self):
        for i in self.models_list:

            if i == "RF":
                self.model_RF.fit(self.X_train, self.y_train)
                self.Res_RF = self.predict_REs(self.X_test,
                                               self.X_train,
                                               self.model_RF)
                self.report(i)
                self.moe_Inputs[i] = self.Res_RF

            elif i == "ET":
                self.model_ET.fit(self.X_train,
                                  self.y_train)
                self.Res_ET = self.predict_REs(self.X_test,
                                               self.X_train,
                                               self.model_ET)
                self.report(i)
                self.moe_Inputs[i] = self.Res_ET
            elif i == "MLP":
                self.model_MLP.fit(self.X_train,
                                   self.y_train)
                self.Res_MLP = self.predict_REs(self.X_test,
                                                self.X_train,
                                                self.model_MLP)
                self.report(i)
                self.moe_Inputs[i] = self.Res_MLP
            elif i == "LR":
                self.model_LR.fit(self.X_train,
                                  self.y_train)
                self.Res_LR = self.predict_REs(self.X_test,
                                               self.X_train,
                                               self.model_LR)
                self.report(i)
                self.moe_Inputs[i] = self.Res_LR


    def predict_REs(self, X_test, X_train,  model):

        train_res = model.predict_proba(X_train)
        test_res = model.predict_proba(X_test)

        return [train_res, test_res]

    def report(self, name):
        print("Module - " + name + " Done!")






















# def prior_mean_field(kernel_size, bias_size, dtype=None):   #prior Function
#     n = kernel_size + bias_size
#     return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype),
#                                                 scale=tf.ones(n)),
#                                      reinterpreted_batch_ndims=1)
#
# def posterior_mean_field(kernel_size, bias_size=0, dtype=None):   #Posterior Function
#   n = kernel_size + bias_size
#   c = np.log(np.expm1(1.))
#   return tf.keras.Sequential([
#       tfp.layers.VariableLayer(2 * n, dtype=dtype),
#       tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#           tfd.Normal(loc=t[..., :n],
#                      scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])),
#           reinterpreted_batch_ndims=1)),
#   ])
#
#
#
# def Model(Data_Path, y):  # Prepair Model
#     try:
#         Data = pd.read_csv(Data_Path)
#     except:
#         Data = pd.read_excel(Data_Path)
#
#     X = Data
#     Y = y
#     columns_name = X.columns
#     ssc = StandardScaler()
#     ssc.fit(X)
#     X = ssc.transform(X)
#     scaler = MinMaxScaler()
#     scaler.fit(X)
#     normalized = scaler.transform(X)
#     X_train1, X_test1, Y_train1, Y_test1 = train_test_split(normalized, y, test_size=0.2, random_state=100)
#
#
#
#     hidden_units = [2, 2, 2]
#     dataset_size = len(Data)
#     batch_size = 50
#     counter_L = 0
#     model = Sequential()
#     InData_Ex1 = Input(shape=(X.shape[1],), name="Input_Ex1")
#     InData_Ex2 = Input(shape=(X.shape[1],), name="Input_EX2")
#     InData_Ex3 = Input(shape=(X.shape[1],), name="Input_EX3")
#     InData_Ex4 = Input(shape=(X.shape[1],), name="Input_EX4")
#     #features = layers.Concatenate([InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4])
#     InData= layers.Concatenate(axis=-1)([InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4])
#     InData = BatchNormalization()(InData)
#     features= InData
#     for units in hidden_units:
#         features = tfp.layers.DenseVariational(
#             units=units,
#             make_prior_fn=prior_mean_field,
#             make_posterior_fn=posterior_mean_field,
#             kl_weight=1 / X.shape[0],
#             activation='relu',
#         )(features)
#     features = layers.Dense(units=1, activation='sigmoid')(features)
#     model = keras.Model(inputs=[InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4], outputs=features)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=['accuracy'],
#     )
#     #history = model.fit(X_train1, Y_train1, validation_data=(X_test1, Y_test1), epochs=500)
#
#     return model, [X_test1, Y_test1], [X_train1, Y_train1]



