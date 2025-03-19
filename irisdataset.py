# Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

class Preprocessor:
    def __init__(self, file):
        self.filepath = file
    
    def read_data(self):
        self.df = pd.read_csv(self.filepath)

    def drop_dup(self):
        self.df = self.df.drop_duplicates().reset_index(drop = True)

    def define_x_y(self):
        x = self.df.drop(columns = "species")
        y = self.df["species"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test

class Modeling:
    def __init__(self, x_train, x_test, y_train, y_test, n_estimators = 100, random_state = 7):
        self.model = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)

    def train(self):
        self.model.fit(x_train, y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(x_test)
        self.model_acc = accuracy_score(y_test, self.y_pred)
        print(f"Accuracy : {self.model_acc}")

    def model_save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

# ----------
preprocessor = Preprocessor("iris.csv")
preprocessor.read_data()
preprocessor.drop_dup()
x_train, x_test, y_train, y_test = preprocessor.define_x_y()

modeling = Modeling(x_train, x_test, y_train, y_test)
modeling.train()
modeling.evaluate()
modeling.model_save("irisdataset.pkl")