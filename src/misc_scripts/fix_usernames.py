import pandas as pd
data = pd.read_csv("refrences/gcj.csv")
split = pd.DataFrame(data['filepath'].str.split("/").tolist())[4]
data['username'] = split
data.to_csv("refrences/gcj.csv")
