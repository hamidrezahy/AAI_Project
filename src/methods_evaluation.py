import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.ensemble import ExtraTreesClassifier # ET
from sklearn.ensemble import AdaBoostClassifier # ADB
from sklearn.svm import SVC # svc
from sklearn.neural_network import MLPClassifier # MLP
from xgboost import XGBModel # XGB
from sklearn.gaussian_process import GaussianProcessClassifier # GPC
from sklearn.naive_bayes import GaussianNB # GNB
from sklearn.linear_model import LogisticRegression # LR
from sklearn.ensemble import GradientBoostingClassifier # GBC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import statistics


class methods:
    def __init__(self, parameters):
        self.datapath = parameters['datapath']
        self.test_size = parameters['test_size']
        self.label = parameters['label']
        self.random_state = parameters['random_state']
        self.models_list = parameters['models_list']
        self. test_num = parameters['test_num']
        self.model_RF = RandomForestClassifier(n_estimators=10,
                                               max_depth=None,
                                               min_samples_split=2,
                                               random_state=0)
        self.model_GNB = GaussianNB(priors=None,
                                    var_smoothing=1e-09)
        self.model_ADB = AdaBoostClassifier(n_estimators=50,
                                            random_state=0)
        self.model_ET = ExtraTreesClassifier(n_estimators=100,
                                             criterion='gini',
                                             min_samples_split=2)
        self.model_GB = GradientBoostingClassifier(n_estimators=100,
                                                   learning_rate=0.1,
                                                   loss='deviance')
        self.model_MLP = MLPClassifier(hidden_layer_sizes=(100,),
                                       activation='relu',
                                       solver='adam',
                                       alpha=0.00001)
        self.model_XGB = XGBModel(random_state=1,
                                  learning_rate=0.5,
                                  n_estimators=7,
                                  maxdepth=5,
                                  eta=0.05,
                                  objective='binary:logistic')
        self.model_LR = LogisticRegression(solver='newtoncg',
                                           C=100)

    def Data_Prepration(self):
        df_input= pd.read_excel(self.datapath)
        Y = df_input.pop(self.label)
        X = df_input
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size,
                                                            random_state=self.random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def models_Prepration(self):
        for i in self.models_list:
            match i:
                case "RF":
                    self.model_RF.fit(self.X_train, self.y_train)
                    self.Res_RF_mean, self.Res_RF_std = self.predict_point(self.X_test,
                                                                           self.model_RF,
                                                                           self.test_num,
                                                                           self.y_test)
                case "GNB":
                    self.model_GNB.fit(self.X_train,
                                       self.y_train)
                    self.Res_GNB_mean, self.Res_GNB_std = self.predict_point(self.X_test,
                                                                             self.model_GNB,
                                                                             self.test_num,
                                                                             self.y_test)
                case "ADB":
                    self.model_ADB.fit(self.X_train,
                                       self.y_train)
                    self.Res_ADB_mean, self.Res_ADB_std = self.predict_point(self.X_test,
                                                                             self.model_ADB,
                                                                             self.test_num,
                                                                             self.y_test)
                case "ET":
                    self.model_ET.fit(self.X_train,
                                      self.y_train)
                    self.Res_ET_mean, self.Res_ET_std = self.predict_point(self.X_test,
                                                                           self.model_ET,
                                                                           self.test_num,
                                                                           self.y_test)
                case "GB":
                    self.model_GB.fit(self.X_train,
                                      self.y_train)
                    self.Res_GB_mean, self.Res_GB_std = self.predict_point(self.X_test,
                                                                           self.model_GB,
                                                                           self.test_num,
                                                                           self.y_test)
                case "MLP":
                    self.model_MLP.fit(self.X_train,
                                       self.y_train)
                    self.Res_MLP_mean, self.Res_MLP_std = self.predict_point(self.X_test,
                                                                            self.model_MLP,
                                                                            self.test_num,
                                                                            self.y_test)
                case "XGB":
                    self.model_XGB.fit(self.X_train,
                                       self.y_train)
                    self.Res_XGB_mean, self.Res_XGB_std = self.predict_point(self.X_test,
                                                                            self.model_XGB,
                                                                            self.test_num,
                                                                            self.y_test)
                case "LR":
                    self.model_LR.fit(self.X_train,
                                      self.y_train)
                    self.Res_LR_mean, self.Res_LR_std = self.predict_point(self.X_test,
                                                                           self.model_LR,
                                                                           self.test_num,
                                                                           self.y_test)




    def predict_point(self, X, model, num_samples, y_test):   #Predict Test Data
      Ou__=[]
      ac_A_Mi=[]
      ac_A_Ma=[]
      ac_A_Bi=[]
      ac_A_We=[]
      pr_A_Mi=[]
      pr_A_Ma=[]
      pr_A_Bi=[]
      pr_A_We=[]
      re_A_Mi=[]
      re_A_Ma=[]
      re_A_Bi=[]
      re_A_We=[]
      f1_A_Mi=[]
      f1_A_Ma=[]
      f1_A_Bi=[]
      f1_A_We=[]
      Mode=[]
      for i in range(num_samples):
        P_ = model.predict(X)
        AC_ = accuracy_score(P_, y_test)
        F1, precision, recall = self.PRF(P_, y_test)
        ac_A_Mi.append(AC_)
        pr_A_Mi.append(precision[0])
        pr_A_Ma.append(precision[1])
        pr_A_Bi.append(precision[2])
        pr_A_We.append(precision[3])
        re_A_Mi.append(recall[0])
        re_A_Ma.append(recall[1])
        re_A_Bi.append(recall[2])
        re_A_We.append(recall[3])
        f1_A_Mi.append(F1[0])
        f1_A_Ma.append(F1[1])
        f1_A_Bi.append(F1[2])
        f1_A_We.append(F1[3])
      LI__= [ac_A_Mi,
            pr_A_Mi,
            pr_A_Ma,
            pr_A_Bi,
            pr_A_We,
            re_A_Mi,
            re_A_Ma,
            re_A_Bi,
            re_A_We,
            f1_A_Mi,
            f1_A_Ma,
            f1_A_Bi,
            f1_A_We]
      ou_nps= []
      ou_npm= []
      for k in LI__:
        ou_nps.append(statistics.stdev(k))

      for k in LI__:
        ou_npm.append(statistics.mean(k))
      return(ou_npm, ou_nps)

    def PRF(self, X__, y_test):  # Calculate precision/ Recall/ F1 Score
        F1 = []
        precision = []
        recall = []
        for i in ['macro', 'micro', 'weighted', 'binary']:
            F1.append(f1_score(y_test, X__, average=i))
            precision.append(precision_score(y_test, X__, average=i))
            recall.append(recall_score(y_test, X__, average=i))
        return F1, precision, recall