import json
# import src.methods_evaluation as mev
#
# Config_Path= 'Config/models_evaluation.json'
# with open(Config_Path) as config_file:
#     cfg = json.load(config_file)
# methods_evl= mev.methods(cfg)
# methods_evl.Data_Prepration()



import os
#import src.MOBE_Main as MOBE_Main

import src.methods_evaluation as me
import src.experience as exp

Config_Path = 'Config/models_evaluation.json'
with open(Config_Path) as config_file:
    cfg = json.load(config_file)
HP = exp.create_experiments()
eval_m = me.methods(cfg)
eval_m.models_Prepration()



# import pandas as pd
#
# df = pd.read_excel('data/Z-Alizadeh sani dataset_preprocessed.xlsx')
# print(df.dtypes)

# heart = MOBE_Main.MOBE('./Config/main_config.json')
#
# heart.Classic_Classifiers(HP)       #Run Classic Methods
#
# heart.MCD_Classifier(HP)        #Run Monte Carlo Dropout Model
#
# heart.BNN_Classifier(HP)        #Run BNN Model
#
# heart.MOE_Classifier(HP)        #Run Proposed Model