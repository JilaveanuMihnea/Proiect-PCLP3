import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset.csv')
dfp = pd.read_csv('dataset_processed.csv')

# Analizam valorile lipsa
print("Missing values in original dataset:")
print(df.isnull().sum())
print("\nAs percentages:")
print(df.isnull().mean() * 100)
print("\nMissing values in processed dataset:")
print(dfp.isnull().sum())

input("\nPress Enter to continue...")

print("\nStatistical summary of dataset:")
print(dfp.describe())
print(dfp.describe(include=[bool]))

input("\nPress Enter to continue...")

# Histogramă pentru fiecare caracteristică numerică
dfp.select_dtypes(include=[np.number]).hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.tight_layout()
plt.show()

# Grafice de tip countplot/barplot pentru variabilele categorice.
for column in dfp.select_dtypes(include=[bool]).columns:
	plt.figure(figsize=(10, 6))
	sns.countplot(data=dfp, x=column)
	plt.title(f'Countplot of {column}')
	plt.xlabel(column)
	plt.ylabel('Count')
	plt.xticks(rotation=45)
	plt.show()
 
# Heatmap pentru a vizualiza corelațiile dintre variabilele numerice
plt.figure(figsize=(12, 8))
corr = dfp.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()
# Boxplot pentru a identifica outliers
for column in dfp.select_dtypes(include=[np.number]).columns:
	if column == 'gen':
		continue
	plt.figure(figsize=(10, 6))
	sns.boxplot(x=dfp[column])
	plt.title(f'Boxplot of {column}')
	plt.xlabel(column)
	plt.show()
# Boxplot pentru variabilele categorice
for column in dfp.select_dtypes(include=['object']).columns:
	plt.figure(figsize=(10, 6))
	sns.boxplot(x=dfp[column])
	plt.title(f'Boxplot of {column}')
	plt.xlabel(column)
	plt.show()

# violin plots pentru relatia
# dintre caracteristici si variabila tinta "durata_somn"
for column in dfp.select_dtypes(include=[np.number]).columns:
	if column == 'durata_somn':
		continue
	binned = pd.cut(dfp[column], bins=10, labels=False)
	plt.figure(figsize=(10, 6))
	sns.violinplot(x=binned, y=dfp['durata_somn'])
	plt.title(f'Violin plot of {column} vs durata_somn')
	plt.xlabel(column)
	plt.ylabel('durata_somn')
	plt.show()
