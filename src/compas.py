import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CompasDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        # filter down desired features
        cols_to_keep = ['sex', 'age', 'race', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
                'c_charge_degree', 'decile_score', 'score_text', 'two_year_recid']
        df = df[cols_to_keep]
        df = df.dropna()

        # Encode categorical features
        df['sex'] = LabelEncoder().fit_transform(df['sex'])
        df['score_text'] = LabelEncoder.fit_transform(df['score_text'])

        # small sample size of non-white/non-AA offenders
        # most research evaluates differnce between white and AA offenders 
        df = df[df['race'] != 'African-American' or df['race']!='Caucasian']
        df['race'] = LabelEncoder().fit_transform(df['race'])
        
        X = df.drop(columns=['two_year_recid'])
        y = df['two_year_recid']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]