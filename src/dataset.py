import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, dataset_path = None):
        self.dataset_path = dataset_path if dataset_path else os.path.join('..', 'data', 'processed', 'dataset.csv')
        self.dataset = pd.read_csv(self.dataset_path)
        self.features, self.labels = None, None
        self.f_train, self.f_test, self.l_train, self.l_test = None, None, None, None
    
    def split_data(self, target_column, test_size=0.2, random_state=42, shuffle=True):
        self.features = self.dataset.drop(columns=[target_column])
        self.labels = self.dataset[target_column]
        self.f_train, self.f_test, self.l_train, self.l_test = train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        return self.f_train, self.f_test, self.l_train, self.l_test