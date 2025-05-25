import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def trainModels(X_train, y_train):
	# Antrenam un model de regresie liniar
	model_lr = LinearRegression()
	model_lr.fit(X_train.select_dtypes(include=[np.number]), y_train)

	# Antrenam un model de regresie cu padure aleatoare
	model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
	model_rf.fit(X_train.select_dtypes(include=[np.number]), y_train)

	# Antrenam un model de regresie cu suport vectorial
	model_svr = SVR(kernel='linear')
	model_svr.fit(X_train.select_dtypes(include=[np.number]), y_train)

	return model_lr, model_rf, model_svr

def compareModels():
	# Incarcam datele prelucrate
	dfp = pd.read_csv('dataset_processed.csv')
 
	# Impartim detele in set de antrenament si set de test
	X = dfp.drop(columns=['durata_somn'])
	y = dfp['durata_somn']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Antrenam modelele
	model_lr, model_rf, model_svr = trainModels(X_train, y_train)

	# Predictii pe setul de test
	y_pred_lr = model_lr.predict(X_test.select_dtypes(include=[np.number]))
	y_pred_rf = model_rf.predict(X_test.select_dtypes(include=[np.number]))
	y_pred_svr = model_svr.predict(X_test.select_dtypes(include=[np.number]))

	# Evaluam modelele
	rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
	mae_lr = mean_absolute_error(y_test, y_pred_lr)
	r2_lr = r2_score(y_test, y_pred_lr)
	rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
	mae_rf = mean_absolute_error(y_test, y_pred_rf)
	r2_rf = r2_score(y_test, y_pred_rf)
	rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
	mae_svr = mean_absolute_error(y_test, y_pred_svr)
	r2_svr = r2_score(y_test, y_pred_svr)

	# Comparam performamta modelelor
	results = pd.DataFrame({
		'Model': ['Linear Regression', 'Random Forest', 'Support Vector Regression'],
		'RMSE': [rmse_lr, rmse_rf, rmse_svr],
		'MAE': [mae_lr, mae_rf, mae_svr],
		'R²': [r2_lr, r2_rf, r2_svr]
	})
	print("\nModel Comparison Results:")
	print(results)

	# Vizualizam rezultatele cu scatter plots
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 3, 1)
	sns.scatterplot(x=y_test, y=y_pred_lr)
	plt.title('Linear Regression Predictions')
	plt.xlabel('Actual Duration of Sleep')
	plt.ylabel('Predicted Duration of Sleep')
	plt.subplot(1, 3, 2)
	sns.scatterplot(x=y_test, y=y_pred_rf)
	plt.title('Random Forest Predictions')
	plt.xlabel('Actual Duration of Sleep')
	plt.ylabel('Predicted Duration of Sleep')
	plt.subplot(1, 3, 3)
	sns.scatterplot(x=y_test, y=y_pred_svr)
	plt.title('Support Vector Regression Predictions')
	plt.xlabel('Actual Duration of Sleep')
	plt.ylabel('Predicted Duration of Sleep')
	plt.tight_layout()
	plt.show()

	# Visualizam distributia erorilor pentru fiecare model
	plt.figure(figsize=(12, 6))
	sns.histplot(y_test - y_pred_lr, kde=True, color='blue', label='Linear Regression', stat='density')
	sns.histplot(y_test - y_pred_rf, kde=True, color='green', label='Random Forest', stat='density')
	sns.histplot(y_test - y_pred_svr, kde=True, color='red', label='Support Vector Regression', stat='density')
	plt.title('Distribution of Errors')
	plt.xlabel('Error (Actual - Predicted)')
	plt.ylabel('Density')
	plt.legend()
	plt.show()

# compareModels()