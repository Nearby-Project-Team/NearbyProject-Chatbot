import pandas as pd
import json
import os 
import glob

DATA_PATH = "../ChatData/PlainConversation"
SNS_LIST = os.listdir(DATA_PATH)

if __name__ == "__main__":

    data_dir_list = []
    for sns in SNS_LIST:
        json_list = glob.glob(DATA_PATH + "/" + sns)
        data_dir_list.extend(json_list)
    
    for d in data_dir_list:
        data = open(d, "r")
        dataJson = json.load(data)
            
    
    