import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.svm import SVR

class feature_selection:
    def __init__(self, parameters):
        self.datapath = parameters['datapath']
        self.label = parameters['label']
        self.methods_list = parameters['methods_list']
        self.CHI2 = SelectKBest(score_func=chi2, k=20)
        self.RFECV = RFECV(SVR(kernel="linear"), step=1, cv=20)
        self.XGB = XGBClassifier()
        self.XGBfeature_length = 32
        self.features = {}

    def data_import(self):
        df_input = pd.read_excel(self.datapath)
        self.Y = df_input.pop(self.label)
        self.X = df_input

    def methods_run(self):
        for i in self.models_list:
            match i:
                case "CHI2":
                    features = self.CHI2.fit(self.X, self.y)
                    self.features['CHI2'] = features
                case "RFECV":
                    features = self.RFECV.fit(self.X, self.y)
                    features = list((self.X.columns[features.get_support()]))
                    self.features['RFECV'] = features
                case "XGB":
                    features = self.XGB.fit(self.X, self.y)
                    f_list = features.feature_importances_
                    selected_features = []
                    counter = 0
                    while(self.XGBfeature_length==counter):
                        f_index = [i for i, j in enumerate(f_list) if j == max(f_list)]
                        counter+=len(f_index)
                        for i in f_index:
                            selected_features.append(f_list[i])
                            f_list[i] = 0

        return self.features


