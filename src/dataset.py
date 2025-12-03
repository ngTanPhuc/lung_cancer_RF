import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self):
        self.dataset_path = os.path.join('..', 'data', 'processed', 'dataseter.csv')
        self.dataset = pd.read_csv(self.dataset_path)
        self.features = None
        self.labels = None
    
    def split_data(self, target_column, test_size=0.2, random_state=42, shuffle=True):
        self.features = self.dataset.drop(columns=[target_column])
        self.labels = self.dataset[target_column]
        f_train, f_test, l_train, l_test = train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        return f_train, f_test, l_train, l_test