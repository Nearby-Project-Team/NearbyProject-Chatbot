import pandas as pd
import json
import os 
import glob
from tqdm import tqdm

DATA_PATH = "../ChatData/PlainConversation"
SNS_LIST = os.listdir(DATA_PATH)

if __name__ == "__main__":

    data_dir_list = []
    for sns in SNS_LIST:
        json_list = glob.glob(DATA_PATH + "/" + sns + "/*.json")
        data_dir_list.extend(json_list)
    
    dataCSV = {
        "Q": [],
        "A": [],
        "label": []
    }
    
    for d in tqdm(data_dir_list):
        data = open(d, "r")
        dataJson = json.load(data)
        conversation = dataJson["info"][0]["annotations"]["lines"]
        N = len(conversation)
        for i in range(1, N):
            pretxt = conversation[i - 1]["norm_text"] 
            txt = conversation[i]["norm_text"]
            dataCSV["Q"].append(pretxt)
            dataCSV["A"].append(txt)
            dataCSV["label"].append(0)
    
    df = pd.DataFrame(dataCSV)
    df.to_csv("../ChatData/chatbot_QA_data_2.csv", index=False)
    
    
            
    
    