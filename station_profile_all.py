"""
    station_profile_all.py

    Train StationProfileAll model
"""
import shelve
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

train_data = "../data/train.csv"
test_data = "../data/test.csv"
model_name = "station_profile_all_svm"
dbfile = "../data/models"  # shelve database file.

data = pd.read_csv(train_data)

# Encode station with integers
with shelve.open(dbfile) as db:
    le = db['labelEncoder']
data.DepStation = le.transform(data.DepStation)
data.ArrStation = le.transform(data.ArrStation)

# Subsample the data
data = data.sample(frac=.128)

X = data[["UserID", "DepHour",  "DepStation", "DepWeek", "DepMonth", "DepMin"]]
y = data["ArrStation"]

# Type conversion. Just to avoid warnings.
X.DepHour = X.DepHour.astype(float)
X.DepStation = X.DepStation.astype(float)
X.DepWeek = X.DepWeek.astype(float)
X.DepMonth = X.DepMonth.astype(float)
X.DepMin = X.DepMin.astype(float)
y = y.astype(float)

# Grid search parameters
C_range = 4. ** np.arange(-3, 8)
gamma_range = 4. ** np.arange(-8, 2)
param_grid = dict(gamma=gamma_range, C=C_range)

scaler = StandardScaler()
grid = GridSearchCV(SVC(class_weight="balanced"), param_grid=param_grid, cv=5, n_jobs=-1, error_score=0)
pipe = make_pipeline(scaler, grid)
model = pipe
model.fit(X, y)

# Save the model
with shelve.open(dbfile) as db:
    db[model_name] = model


# Test phase
data = pd.read_csv(test_data)
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


