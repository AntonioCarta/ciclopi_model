"""
    user_profile.py

    Train the UserProfile model
"""
import shelve
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from user_specific_model import UserSpecializedEstimator
from sklearn.metrics import classification_report

train_data = "../data/train.csv"
model_name = "user_profile_svm"
dbfile = "../data/models"  # shelve database file.

data = pd.read_csv(train_data)

# Encode station names with integers
with shelve.open(dbfile) as db:
    le = db['labelEncoder']
data.DepStation = le.transform(data.DepStation)
data.ArrStation = le.transform(data.ArrStation)

X = data[["UserID", "DepHour",  "DepStation", "DepWeek", "DepMonth", "DepMin"]]
y = data["ArrStation"]

X.DepHour = X.DepHour.astype(float)
X.DepStation = X.DepStation.astype(float)
X.DepWeek = X.DepWeek.astype(float)
X.DepMonth = X.DepMonth.astype(float)
X.DepMin = X.DepMin.astype(float)
y = y.astype(float)

C_range = 4. ** np.arange(-3, 8)
gamma_range = 4. ** np.arange(-8, 2)
param_grid = dict(gamma=gamma_range, C=C_range)
scaler = StandardScaler()
grid = GridSearchCV(SVC(class_weight="balanced"), param_grid=param_grid, cv=5, n_jobs=-1, error_score=0)
pipe = make_pipeline(scaler, grid)

model = UserSpecializedEstimator(pipe, "UserID", 20)
model.fit(X, y)

with shelve.open(dbfile) as db:
    db[model_name] = model

data = pd.read_csv("../data/test.csv")
data.DepStation = le.transform(data.DepStation)
data.ArrStation = le.transform(data.ArrStation)

X = data[["UserID", "DepHour",  "DepStation", "DepWeek", "DepMonth", "DepMin"]]
y = data["ArrStation"]

X.DepHour = X.DepHour.astype(float)
X.DepStation = X.DepStation.astype(float)
X.DepWeek = X.DepWeek.astype(float)
X.DepMonth = X.DepMonth.astype(float)
X.DepMin = X.DepMin.astype(float)
y = y.astype(float)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print("accuracy: " + str(acc))

s = classification_report(y, y_pred, target_names=le.classes_)
print(s)
