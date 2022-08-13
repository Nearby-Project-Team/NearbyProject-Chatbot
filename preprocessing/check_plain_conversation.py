import pandas as pd

data = pd.read_csv("../ChatData/chatbot_QA_data_2.csv")

for i, row in data.iterrows():
    q = row["Q"]
    a = row["A"]
    if (type(q) != type("")) or (type(a) != type("")):
        print(q, type(q))
        print(a, type(a))
        print(i) 