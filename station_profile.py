"""
    station_profile_all.py

    Train the StationProfile model.
"""
import shelve
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time

train_data = "../data/train.csv"
model_name = "model_station_profile_all_svm"
dbfile = "../data/models"  # shelve database file.

data = pd.read_csv(train_data)

# Encode stations with integers.
with shelve.open(dbfile) as db:
    le = db['labelEncoder']
data.DepStation = le.transform(data.DepStation)
data.ArrStation = le.transform(data.ArrStation)

# dictionary of models for each station
models = {}

# training times
training_times = []
training_times.append(time.asctime())

# Train phase
for staz in range(15):
    # Select the data for the current station
    sel_data = data[data.DepStation == staz]

    X = sel_data[["UserID", "DepHour",  "DepStation", "DepWeek", "DepMonth", "DepMin"]]
    y = sel_data["ArrStation"]

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
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid=param_grid, cv=5, n_jobs=-1, error_score=0)
    pipe = make_pipeline(scaler, grid)

    model = pipe
    model.fit(X, y)
    models[staz] = model

    # save time
    training_times.append(time.asctime())

# Save model
with shelve.open(dbfile) as db:
    db[model_name] = models

data = pd.read_csv("../data/test.csv")
data.DepStation = le.transform(data.DepStation)
data.ArrStation = le.transform(data.ArrStation)

# Test phase
y_tot, y_tot_pred = [], []
for staz in range(15):
    sel_data = data[data.DepStation == staz]
    X = sel_data[["UserID", "DepHour",  "DepStation", "DepWeek", "DepMonth", "DepMin"]]
    y = sel_data["ArrStation"]

    X.DepHour = X.DepHour.astype(float)
    X.DepStation = X.DepStation.astype(float)
    X.DepWeek = X.DepWeek.astype(float)
    X.DepMonth = X.DepMonth.astype(float)
    X.DepMin = X.DepMin.astype(float)
    y = y.astype(float)

    model = models[staz]
    y_pred = model.predict(X)

    # Union of prediction
    for el in y:
        y_tot.append(el)
    for el in y_pred:
        y_tot_pred.append(el)

acc = accuracy_score(y_tot, y_tot_pred)
print("accuracy: " + str(acc))

s = classification_report(y_tot, y_tot_pred, target_names=le.classes_)
print(s)





