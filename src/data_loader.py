import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
    def load_and_split(self):
        try:
            df = pd.read_csv(self.filepath)
            X = df.drop('critical_temp', axis=1) # Independant variable
            y = df['critical_temp'] # Dependant variable  
            return train_test_split(X, y, test_size=0.2, random_state=100)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file at {self.filepath}")