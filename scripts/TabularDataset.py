# TabularDataset.py

import pandas as pd
import re
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, csv_file1, csv_file2, target_column, transform=None):
        self.data1 = pd.read_csv(csv_file1)
        self.data2 = pd.read_csv(csv_file2)

        # Keep only the relevant columns including 'image_id'
        self.data1 = self.keep_relevant_columns(self.data1)
        self.data2 = self.keep_relevant_columns(self.data2)

        # Merge the two datasets on 'image_id'
        self.data = pd.merge(self.data1, self.data2, on='image_id')

        # Ensure the target column exists in the merged dataset
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the merged dataset.")

        # Extract target and features
        self.target = self.data[target_column].values
        self.features = self.data.drop(columns=[target_column, 'image_id']).values
        self.transform = transform

    def keep_relevant_columns(self, dataframe):
        """Keep only columns that are named 'dx', 'image_id', start with 'original_', or contain only numeric characters."""
        relevant_columns = ['image_id']  # Always keep 'image_id'
        for col in dataframe.columns:
            if col == 'dx':
                relevant_columns.append(col)
            elif col.startswith('original_'):
                relevant_columns.append(col)
            elif re.fullmatch(r'\d+', col):
                relevant_columns.append(col)
        return dataframe[relevant_columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, target
