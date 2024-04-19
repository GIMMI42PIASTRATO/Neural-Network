import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

labelEncoder = LabelEncoder()

data = pd.read_csv("data\\Cancer_Data.csv")
print(data.shape)
# print("--------------------")
X = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
print(X)

# print("--------------------")
# Y = data["diagnosis"]
# print(labelEncoder.fit_transform(Y))
