from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("data/raw/CEE_DATA.arff")
df = pd.DataFrame(data)
df.to_csv("data/raw/CEE_DATA.csv", index=False)
print(df.shape)