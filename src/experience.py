import os
import src.MOBE_Main as MOBE_Main
import src.experiments as exp

HP= exp.create_experiments()

heart= MOBE_Main.MOBE('./Config/main_config.json')

heart.Classic_Classifiers(HP)       #Run Classic Methods

heart.MCD_Classifier(HP)        #Run Monte Carlo Dropout Model

heart.BNN_Classifier(HP)        #Run BNN Model

heart.MOE_Classifier(HP)        #Run Proposed Model