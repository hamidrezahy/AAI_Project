import json
import src.methods_evaluation as mev

Config_Path= 'Config/models_evaluation.json'
with open(Config_Path) as config_file:
    cfg = json.load(config_file)
methods_evl= mev.methods(cfg)
methods_evl.Data_Prepration()

