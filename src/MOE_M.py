import os
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.ensemble import ExtraTreesClassifier # ET
from sklearn.neural_network import MLPClassifier # MLP
from sklearn.linear_model import LogisticRegression # LR
from sklearn.model_selection import KFold
import statistics
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dropout, BatchNormalization
np.set_printoptions(precision=3, suppress=True)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

tfd = tfp.distributions

class moe:
    def __init__(self, parameters, HP, dataset_name):
        self.datapath = parameters[dataset_name]
        self.resname= dataset_name+'_BNN_moe'
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
        # ssc = StandardScaler()
        # ssc.fit(train_res)
        # train_res = ssc.transform(train_res)
        # ssc = StandardScaler()
        # ssc.fit(test_res)
        # test_res = ssc.transform(test_res)

        return [train_res, test_res]

    def report(self, name):
        print("Module - " + name + " Done!")


    def prior_mean_field(self, kernel_size, bias_size, dtype=None):   #prior Function
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype),
                                                    scale=tf.ones(n)),
                                        reinterpreted_batch_ndims=1)

    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):   #Posterior Function
      n = kernel_size + bias_size
      c = np.log(np.expm1(1.))
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])),
              reinterpreted_batch_ndims=1)),
      ])

    def prior(self, kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    def posterior(self, kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    # def priorED1(self, kernel_size, bias_size, dtype = None):
    
    #     n = kernel_size + bias_size
    
    #     prior_model = Sequential([tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc = tf.zeros(n)  ,  scale_diag = tf.ones(n)))])
        
    #     return(prior_model)

    # def posteriorED1(self, kernel_size, bias_size, dtype = None):
        
    #     n = kernel_size + bias_size
        
    #     posterior_model = Sequential([
            
    #         tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n)  , dtype = dtype),tfp.layers.MultivariateNormalTriL(n)])
        
    #     return(posterior_model)
    # def negative_loglikelihood(self, targets, estimated_distribution):
    #     return -estimated_distribution.log_prob(targets)
    # noise = 1.0
    # def negative_loglikelihood(self, y_obs, y_pred, sigma=noise):
    #    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    #    return keras.backend.sum(-dist.log_prob(tf.cast(y_obs, tf.float32)))
    # def negative_loglikelihood(self, targets, estimated_distribution):
    #     return -estimated_distribution.log_prob(tf.cast(targets, tf.float32))
    def Model(self):
        self.models_Prepration()
        hidden_units = [2, 2]
        batch_size = 50
        counter_L = 0
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(1, dtype=tf.float32), scale=1.0), reinterpreted_batch_ndims=1)
        neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
        model = Sequential()
        InData_Ex1 = Input(shape=(self.moe_Inputs['RF'][0].shape[1],), name="Input_Ex1")
        InData_Ex2 = Input(shape=(self.moe_Inputs['ET'][0].shape[1],), name="Input_EX2")
        InData_Ex3 = Input(shape=(self.moe_Inputs['MLP'][0].shape[1],), name="Input_EX3")
        InData_Ex4 = Input(shape=(self.moe_Inputs['LR'][0].shape[1],), name="Input_EX4")
        InData_Ex5 = Input(shape=(self.X_train.shape[1],), name="in_Data")
        #features = layers.Concatenate([InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4])
        InData= layers.Concatenate(axis=-1)([InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4, InData_Ex5])
        InData = BatchNormalization()(InData)
        features= InData
        # features = layers.Dense(units=8, activation='relu')(features)
        # features = layers.Dense(units=8, activation='relu')(features)
        # features = layers.Dense(units=8, activation='signoid')(features)
        # features = layers.Dense(10, activation="relu")(features)
        # features = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(1), activation=None)(features)
        # features = tfp.layers.MultivariateNormalTriL(1, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/10))(features)
        for units in hidden_units:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn= self.prior_mean_field,
                make_posterior_fn= self.posterior_mean_field,
                kl_weight=1 / 31,
                activation='tanh',
            )(features)
            # features = layers.Dense(units=20, activation='sigmoid')(features)
            # features= layers.Dropout(.2)(features)

        # features = layers.Dense(10, activation="relu")(features)


        # features = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(1), activation=None)(features)
        # features = tfp.layers.MultivariateNormalTriL(1, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/10))(features)
        ##features = layers.Dense(units=1, activation='sigmoid')(features)
        features = layers.Dense(units=2)(features)
        features= tf.keras.layers.Dense(1)(features)
        features= tfp.layers.IndependentBernoulli(1)(features)
        # features = layers.Dense(units=1, activation= 'softmax')(features)
        #features = layers.Dense(tfp.layers.IndependentBernoulli.params_size(1)(features))
        #features = tfp.layers.IndependentBernoulli(1)(features)
        #distribution_params = layers.Dense(units=2)(features)  
        #outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        model = keras.Model(inputs=[InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4, InData_Ex5], outputs=features)
        model.compile(
            optimizer= keras.optimizers.Adam(learning_rate=0.001),
            loss= neg_log_likelihood,
            metrics=['accuracy'])
        print(model.summary())
        # history = model.fit([self.moe_Inputs['RF'][0],
        #                     self.moe_Inputs['ET'][0],
        #                     self.moe_Inputs['MLP'][0],
        #                     self.moe_Inputs['LR'][0]],
        #                     self.y_train,
        #                     epochs=150,
        #                     validation_data=([self.moe_Inputs['RF'][1],
        #                                       self.moe_Inputs['ET'][1],
        #                                       self.moe_Inputs['MLP'][1],
        #                                       self.moe_Inputs['LR'][1]],
        #                                       self.y_test),
        #                                       batch_size=64)
        self.moe_model= model
    
    def MOE_Run(self):
        self.Model()
        moe_acc=[]
        history= self.moe_model.fit([self.moe_Inputs['RF'][0],
                                    self.moe_Inputs['ET'][0],
                                    self.moe_Inputs['MLP'][0],
                                    self.moe_Inputs['LR'][0],
                                    self.X_train],
                                    self.y_train,
                                    epochs=1000,
                                    validation_data=([self.moe_Inputs['RF'][1],
                                                      self.moe_Inputs['ET'][1],
                                                      self.moe_Inputs['MLP'][1],
                                                      self.moe_Inputs['LR'][1],
                                                      self.X_test],
                                                      self.y_test),
                                                      batch_size=64)

        # final_result= []
        # for i in  predicted:
        #   if i>=0.5:
        #     final_result.append(1)
        #   else:
        #     final_result.append(0)
        acc_=[]
        precision_=[]
        recall_=[]
        f1_=[]
        res_dic = {}
        for i in range(100):
          predicted= self.moe_model.predict([self.moe_Inputs['RF'][1],
                                            self.moe_Inputs['ET'][1],
                                            self.moe_Inputs['MLP'][1],
                                            self.moe_Inputs['LR'][1],
                                            self.X_test])
          acc_.append(accuracy_score(self.y_test, predicted))
          precision_.append(precision_score(self.y_test, predicted))
          recall_.append(recall_score(self.y_test, predicted))
          f1_.append(f1_score(self.y_test, predicted))
        res_dic["acc"]= [str(np.mean(acc_))[:4]+' ± '+ str(np.std(acc_))[:4]]  
        res_dic["precision"]= [str(np.mean(precision_))[:4]+' ± '+ str(np.std(acc_))[:4]]
        res_dic["recall"]= [str(np.mean(recall_))[:4]+' ± '+ str(np.std(acc_))[:4]]
        res_dic["f1"]= [str(np.mean(f1_))[:4]+' ± '+ str(np.std(acc_))[:4]]
        resdf = pd.DataFrame(res_dic)
        resdf.to_excel(self.experience_path+'/'+self.resname+'.xlsx')

        # print(predicted)
        # for i in range(100):
        #   predicted= self.moe_model.predict([self.moe_Inputs['RF'][1],
        #                                      self.moe_Inputs['ET'][1],
        #                                      self.moe_Inputs['MLP'][1],
        #                                      self.moe_Inputs['LR'][1]])
        #   moe_acc.append(accuracy_score(self.y_test, predicted))
        #   print(moe_acc.mean(), moe_acc.std())
        return history
























