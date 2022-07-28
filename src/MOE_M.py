import os
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
import statistics
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  Input


def prior_mean_field(kernel_size, bias_size, dtype=None):   #prior Function
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype),
                                                scale=tf.ones(n)),
                                     reinterpreted_batch_ndims=1)

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):   #Posterior Function
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])



def RUN(Data_Path, y):  # Prepair Model
    try:
        Data = pd.read_csv(Data_Path)
    except:
        Data = pd.read_excel(Data_Path)

    X = Data
    Y = y
    columns_name = X.columns
    ssc = StandardScaler()
    ssc.fit(X)
    X = ssc.transform(X)
    scaler = MinMaxScaler()
    scaler.fit(X)
    normalized = scaler.transform(X)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(normalized, y, test_size=0.2, random_state=100)
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    X_train1 = quantile_transformer.fit_transform(X_train1)
    X_test1 = quantile_transformer.transform(X_test1)
    X_train1 = pd.DataFrame(data=X_train1, columns=columns_name)
    X_test1 = pd.DataFrame(data=X_test1, columns=columns_name)


    hidden_units = [2, 2, 2]
    dataset_size = len(Data)
    batch_size = 50
    counter_L = 0
    model = Sequential()
    InData = Input(shape=(X.shape[1],), name="Input")
    features = InData
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior_mean_field,
            make_posterior_fn=posterior_mean_field,
            kl_weight=1 / X.shape[0],
            activation='relu',
        )(features)
    features = layers.Dense(units=1, activation='sigmoid')(features)
    model = keras.Model(inputs=InData, outputs=features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', F1_S, PR_S, Re_S],
    )
    history = model.fit(X_train1, Y_train1, validation_data=(X_test1, Y_test1), epochs=500)

    return history, model, [X_test1, Y_test1], [X_train1, Y_train1]




