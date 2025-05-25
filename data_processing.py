import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def processData():
	# Incarcam dataset-ul creat in partea 1 pentru a il prelucra
	dfp = pd.read_csv('dataset.csv');

	# Incepem prelucrarea prin gestionarea valorilor lipsa
	print("Filling missing values...")
	dfp["gen"] = dfp["gen"].replace(np.nan, np.random.choice(dfp["gen"].dropna().unique()))
	dfp["nivel_stres"] = dfp["nivel_stres"].replace(
		np.nan, np.random.choice(dfp["nivel_stres"].dropna().unique(), p=dfp["nivel_stres"].value_counts(normalize=True))
	)
	dfp["ora_culcare"] = dfp["ora_culcare"].apply(lambda x: x + 24 if x < 20 else x)
	meanVal = dfp.mean(numeric_only=True)
	dfp = dfp.fillna(meanVal)
	print("Missing values handled.")

	print("Converting categorical variables to numerical...")
	# One-hot encoding pentru coloana 'nivel_stres'
	dfp = pd.get_dummies(dfp, columns=["nivel_stres"], drop_first=False)
	# Label encoding pentru coloana 'gen'
	le = LabelEncoder()
	dfp["gen"] = le.fit_transform(dfp["gen"])
	print("Categorical variables converted.")

	# Normalizare si standardizare
	print("Normalizing and standardizing numerical features...")
	scaler = MinMaxScaler()
	dfp[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]] = scaler.fit_transform(
		dfp[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]]
	)
	scaler = StandardScaler()
	dfp[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]] = scaler.fit_transform(
		dfp[["varsta", "ore_ecran", "cafea", "minute_sport", "ora_culcare", "zgomot"]]
	)
	print("Normalization and standardization completed.")

	# Salvam dataset-ul prelucrat
	dfp.to_csv('dataset_processed.csv', index=False)

# processData()