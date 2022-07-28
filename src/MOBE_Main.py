import json
import src.MOE_M as MOEM
import matplotlib.pyplot as plt
import os

class MOBE:                #MOBE s
  def __init__(self, Config_Path):
    with open(Config_Path) as config_file:
      cfg = json.load(config_file)tands for Mixture of Bayesian Experts
    self.model_path = cfg["bayasian-model_config_path"]
    self.data_Path = cfg["data_Path"]
    self.Clc_Path_Dic = cfg["Clc_PathDic"]

  def MOE_Classifier(self, Path):         #Mixture of Experts Model (Final Model)
    Clc_path= Path+'/MOE_Classifier_Results'
    if not os.path.exists(Clc_path):
      os.makedirs(Clc_path)
    MOEM.run_model(Clc_path)