import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
# from pandas_profiling import ProfileReport 
 

import sklearn.linear_model
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
#pip install catboost
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
#Dimensionality Reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import xgboost as xgb 

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Imputer
from sklearn.impute import SimpleImputer

%matplotlib inline

df = pd.read_excel('/content/drive/MyDrive/shack_labs/part1.xlsx')


#Checking if there are any null values in any of the columns
names = []
for x in df.isnull().sum().iteritems():
    print(x[0], " ", x[1])

# Thus there are no missing values in the data and we can proceed for further analysis without any imputation


# Correlation matrix between each of the features with one another
df.corr()

# Defining input and output variables as X and y respectively
X = df.drop('House price of unit area', axis = 1)
y = df['House price of unit area']

### Exploratory Data Analysis
#Distribution of target labels
x = df['House price of unit area']
sns.set_style('whitegrid')
sns.distplot(x)
plt.show()

#Correlation matrix plot
# Plotting the correlations in data set 
plt.figure(figsize=(7,7))
sns.heatmap(df.corr())

# Plotting the correlations in data set 
plt.figure(figsize=(7,7))
sns.heatmap(df.corr())

#We observe positive correlation of latitude, longitude, size, number of convenience stores and bedrooms with house price and negative correlation with transaction date and house age. Indicating older houses have negative impact on prices. Both of these seem logical and the correlations do no look suspicious.
for features in list(X.columns):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    x = df[features]
    sns.jointplot(x=x, y=y, data=df)

# Checking if any row is duplicated 
df.duplicated().sum()
sns.pairplot(df)

# Outlier Detection
contf = ['Transaction date', 'House Age', 'Distance from nearest Metro station (km)', 'latitude', 'longitude', 'House size (sqft)', 'House price of unit area']

for i in contf:
  data = df.copy()
  if 0 in data[i].unique():
    pass
  else:
    data[i] = np.log(data[i])
    data.boxplot(column = i)
    plt.ylabel(i)
    plt.show()

# Identifying categorical and numerical input features
df.dtypes

# We observe all features are numerical
df.hist(figsize=(14,14), xrot=20)
plt.show()
sns.countplot(y=df['Number of convenience stores'], data=df)
plt.show()
sns.countplot(y=df['Number of bedrooms'], data=df)
plt.show()


### Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)

# Linear Regression Model
lg = sklearn.linear_model.LinearRegression()
lg.fit(X_train, y_train)

lg_rmse = mean_squared_error(y_test, lg.predict(X_test), squared=False)
lg_rmse



rf = RandomForestRegressor(n_jobs = -1, random_state = 0)
rf.fit(X_train, y_train)

rf_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
rf_rmse


dt = DecisionTreeRegressor(max_depth = 3)
dt.fit(X_train, y_train)

dt_rmse = mean_squared_error(y_test, dt.predict(X_test), squared=False)
dt_rmse


xgb = XGBRegressor(n_estimators=100, max_depth=6, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xgb.fit(X_train, y_train)

xgb_rmse = mean_squared_error(y_test, xgb.predict(X_test), squared=False)
xgb_rmse



cb = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=4)
cb.fit(X_train, y_train)

cb_rmse = mean_squared_error(y_test, cb.predict(X_test), squared=False)
cb_rmse


svr = SVR(kernel='rbf', C=1000, epsilon=1)
svr.fit(X_train, y_train)

svr_rmse = mean_squared_error(y_test, svr.predict(X_test), squared=False)
svr_rmse


knn = KNeighborsRegressor(n_neighbors = 10, weights = "distance")
knn.fit(X_train, y_train)

knn_rmse = mean_squared_error(y_test, knn.predict(X_test), squared=False)
knn_rmse


ridge = Ridge(alpha=0.1, solver="cholesky")
ridge.fit(X_train, y_train)

ridge = mean_squared_error(y_test, ridge.predict(X_test), squared=False)
ridge


lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

lasso = mean_squared_error(y_test, lasso.predict(X_test), squared=False)
lasso


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(X_train_poly, y_train)
poly_lr_rmse = mean_squared_error(y_test, lin_reg.predict(X_test_poly), squared=False)
poly_lr_rmse
#Gradient boosting regressor



gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

gbr_rmse = mean_squared_error(y_test, gbr.predict(X_test), squared=False)
gbr_rmse


# Set parameters for Grid Search
param_grid =  {'n_estimators':[3, 10, 30, 100, 200, 300, 400, 500, 600],
               'max_features':[2, 3, 4, 6, 8]  #0.1, 0.3, 0.6, 
              }
# Initialise the random forest model 
RandForest = RandomForestRegressor(n_jobs= -1, random_state = 0, bootstrap=True)

# Initialise Gridsearch CV with 5 fold corssvalidation and neggative root_mean_squared_error
grid_search = GridSearchCV(estimator=RandForest, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, return_train_score=True)

# Tuned_RandForest.fit(X_train, y_train)
grid_search.fit(X_train, y_train)

grid_search.best_params_
Tuned_RandForest = grid_search.best_estimator_
Tuned_RandForest.fit(X_train, y_train)

mean_squared_error(y_test, Tuned_RandForest.predict(X_test), squared=False)

# Set paramters for Grid Search
param_grid =  {'booster':['gbtree','gblinear'],
               'n_estimators':[3, 10, 30, 100, 200, 300, 400, 500, 600],
               'max_depth':[2, 3, 4, 6, 8],
               'eta':[0.01,0.1,0.3,1]
              }
# Initialise the xgb model 
xgbr = XGBRegressor(subsample=0.7)

# Initialise Gridsearch CV with 5 fold corssvalidation and neggative root_mean_squared_error
grid_search = GridSearchCV(estimator=xgbr, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, return_train_score=True)

# Tuned_RandForest.fit(X_train, y_train)
grid_search.fit(X_train, y_train)

grid_search.best_params_
Tuned_XGB = grid_search.best_estimator_
Tuned_XGB.fit(X_train, y_train)

mean_squared_error(y_test, Tuned_XGB.predict(X_test), squared=False)

# Set paramters for Grid Search
param_grid =  {'depth':[2, 3, 4, 6, 8],
               'learning_rate':[0.01,0.1,0.3,1],
               'l2_leaf_reg':[0.1,0.3,1,3, 10],
               #'n_estimators' : [10, 50, 100, 500],
               'subsample' : [0.5, 0.7, 1.0]
              }
# Initialise the xgb model 
cbr = CatBoostRegressor(iterations=500)

# Initialise Gridsearch CV with 5 fold corssvalidation and neggative root_mean_squared_error
grid_search = GridSearchCV(estimator=cbr, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, return_train_score=True)

grid_search.fit(X_train, y_train)

grid_search.best_params_
Tuned_cb = grid_search.best_estimator_
Tuned_cb.fit(X_train, y_train)

mean_squared_error(y_test, Tuned_cb.predict(X_test), squared=False)


# Set paramters for Grid Search
param_grid =  {'max_depth':[2, 3, 4, 6, 8],
               'learning_rate':[0.01,0.1,1],
               'n_estimators' : [10, 50, 100, 500],
               'subsample' : [0.5, 0.7, 1.0]
              }
# Initialise the xgb model 
gbrg = GradientBoostingRegressor()

# Initialise Gridsearch CV with 5 fold corssvalidation and neggative root_mean_squared_error
grid_search = GridSearchCV(estimator=gbrg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, return_train_score=True)

grid_search.fit(X_train, y_train)

grid_search.best_params_
Tuned_gbr = grid_search.best_estimator_
Tuned_gbr.fit(X_train, y_train)

mean_squared_error(y_test, Tuned_gbr.predict(X_test), squared=False)


a = pd.DataFrame(y_test)
a['cb'] = Tuned_cb.predict(X_test)
a['xgb'] = Tuned_XGB.predict(X_test)
a['rf'] = Tuned_RandForest.predict(X_test)
a['gbr'] = Tuned_gbr.predict(X_test)
a['pred'] = (a['cb'] + a['xgb'] + a['rf'] + a['gbr'])/4

mean_squared_error(y_test, a['pred'], squared=False)

estimators = [
  ('cb', CatBoostRegressor(depth=6, l2_leaf_reg = 0.3, learning_rate = 0.01, subsample = 1.0)),
  ('rf', RandomForestRegressor(max_features=2, n_estimators=200, n_jobs=-1,random_state=0)),
  ('gbr', GradientBoostingRegressor(max_depth=2, n_estimators=50,learning_rate = 0.1)),
  ('xgb', XGBRegressor(eta=0.01, n_estimators=30, subsample=0.7, max_depth=3))
]



model = StackingRegressor(estimators=estimators)

model.fit(X_train,y_train)

rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
r2 = r2_score(y_test, model.predict(X_test))