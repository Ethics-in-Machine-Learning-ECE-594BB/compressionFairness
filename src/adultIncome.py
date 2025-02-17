import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader

class AdultIncomeDataset(Dataset):
    def __init__(self, file_path):
        # Load dataset
        column_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship", "race", "sex",
                        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

        df = pd.read_csv(file_path, names=column_names, skiprows=1)

        # Drop missing values
        df = df.replace(" ?", pd.NA).dropna()

        # Encode categorical features
        categorical_features = ["workclass", "education", "marital-status", "occupation", 
                                "relationship", "race", "sex", "native-country"]

        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])

        # Convert income to binary label
        df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

        # Separate features and labels
        X = df.drop(columns=["income"])
        y = df["income"]

        # Normalize numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
