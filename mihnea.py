import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Incarcam dataframe-ul pentru a il prelucra
df_prel = pd.read_csv('full_dataset.csv');
# print(df_prel);

# Incepem prelucrarea prin gestionarea valorilor lipsa
print("Filling missing values...")

df_prel["gen"] = df_prel["gen"].fillna(df_prel["gen"].mode()[0])
df_prel["nivel_stres"] = df_prel["nivel_stres"].fillna(df_prel["nivel_stres"].mode()[0])
mean_values = df_prel.mean(numeric_only=True)
df_prel = df_prel.fillna(mean_values)
print("Missing values handled.")


print("Converting categorical variables to numerical...")
# One-hot encoding pentru coloana 'nivel_stres'
df_prel = pd.get_dummies(df_prel, columns=["nivel_stres"], drop_first=True)
# Label encoding pentru coloana 'gen'
le = LabelEncoder()
df_prel["gen"] = le.fit_transform(df_prel["gen"])
print("Categorical variables converted.")


# Normalizare si standardizare
print("Normalizing and standardizing numerical features...")
scaler = MinMaxScaler()
df_prel[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]] = scaler.fit_transform(
	df_prel[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]]
)
scaler = StandardScaler()
df_prel[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]] = scaler.fit_transform(
	df_prel[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]]
)
print("Normalization and standardization completed.")



