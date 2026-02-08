import pandas as pd

# load data seat
url="https://github.com/mwaskom/seaborn-data/blob/master/tips.csv"


df =pd.read_csv(url)

print(df)