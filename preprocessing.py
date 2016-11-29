# -*- coding: utf-8 -*-
"""
    preprocessing.py

    Create the train and test set and does some basic data cleaning and feature extraction.
"""
import pandas as pd
import numpy as np
from pandas import Timedelta
from sklearn.cross_validation import ShuffleSplit as ShuffleSplit
import shelve
from sklearn.preprocessing import LabelEncoder

# Load the csv file
df = pd.read_csv(r'../data/fileName.csv.gz', sep=';')

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["dtmDataOraPrelievo"])
df["timestampArr"] = pd.to_datetime(df["dtmDataOraRiconsegna"])
df["date"] = df.timestamp.apply(lambda x: x.date())

# Eliminate small pickups
delta = Timedelta('0 days 00:01:00')
df = df[(df.strNome != df.Expr1) | (df["timestampArr"] - df["timestamp"] > delta)]


# Remove useless features
df = df.drop(['intColonninaPrelievo', 'intColonninaDeposito',
              'intColonninaDeposito', 'dtmDataOraPrelievo', 'dtmDataOraPrelievo',
              'dtmDataOraRiconsegna'], axis=1)


# Rename features
df = df.rename(columns={'strNome': 'DepStation', 'Expr1': 'ArrStation', 'Utente': 'UserID'})

# Remove some stations outside of Pisa.
arrivalStation = df.groupby(["ArrStation"]).size()
farawayStation = arrivalStation[arrivalStation < 15].index
df = df[~(df.ArrStation.isin(farawayStation) | df.DepStation.isin(farawayStation))]

# Extract time features.
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["DepHour"] = df["timestamp"].apply(lambda x: x.hour)
df["DepMin"] = df["timestamp"].apply(lambda x: x.minute)
df["DepWeek"] = df["timestamp"].apply(lambda x: x.weekday())
df["DepMonth"] = df["timestamp"].apply(lambda x: x.month)

# Eliminate users with an unusually high number of pickups
gb = df.groupby("UserID").size()
gb = gb[gb < 2000]
df = df[df.UserID.isin(gb.index)]

# Divide train and test set
days = pd.Series(df.date.unique())
ss = ShuffleSplit(len(days), n_iter=1, test_size=0.3)
for train_index, test_index in ss:
    train_idx = (df.date.isin(days.iloc[train_index]))
    test_idx = (df.date.isin(days.iloc[test_index]))

    X_train, X_test = df[train_idx], df[test_idx]


X_train.to_csv('../data/train.csv')
X_test.to_csv('../data/test.csv')


# Convert station name to an integer
le = LabelEncoder()
le.fit(X_train.DepStation)
with shelve.open("../data/models") as db:
    db['labelEncoder'] = le


