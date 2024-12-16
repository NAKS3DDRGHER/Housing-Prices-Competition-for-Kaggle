import pandas as pd
import numpy as np
from linear_reg_model import LinearRegress
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def  z_score_normalization(x_train, mu, sigma):
    for i in range(x_train.shape[1]):
        if sigma[i] != 0:
            x_train[:,i] = (x_train[:,i] - mu[i]) / sigma[i]
    return x_train

# upload data
train = pd.read_csv('data/n_train.csv').to_numpy()
labels = pd.read_csv('data/labels.csv').to_numpy()
test = pd.read_csv('data/n_test.csv')
id_test = pd.DataFrame(test['Id'])
test = test.drop('Id', axis=1)

# feature scaling
mu = np.mean(train, axis=0)
sigma = np.std(train, axis=0)
norm_train = z_score_normalization(train, mu, sigma)
norm_test = z_score_normalization(test.to_numpy(), mu, sigma)


labels = labels.flatten()
log_labels = np.log(labels)

# custom lineal regression
# model = LinearRegress(norm_train, labels)
# model.fit()

# solution with sklearn
# # model = LinearRegression()
model = RandomForestRegressor(random_state=0)
# # model = DecisionTreeRegressor(random_state=0)
# model.fit(norm_train, labels)


ans = model.predict(norm_test)
id_test["SalePrice"] = ans
id_test.to_csv("data/ans.csv", index=False)

# print(model.score(norm_train, labels))
