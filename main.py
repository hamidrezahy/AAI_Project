import os
import src.experience as exp
import src.MOBE_Main as MOBE_Main

HP= exp.create_experiments()

heart= MOBE_Main.MOBE('./Config/main_config.json')


heart.MOE_Classifier(HP)        #Run Proposed Model