import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dfp = pd.read_csv('dataset_processed.csv')
# Împărțim datele în seturi de antrenament și test
X = dfp.drop(columns=['durata_somn'])
y = dfp['durata_somn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenăm un model de regresie liniară
model_lr = LinearRegression()
model_lr.fit(X_train.select_dtypes(include=[np.number]), y_train)
# Antrenăm un model de regresie cu pădure aleatoare
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train.select_dtypes(include=[np.number]), y_train)
# Antrenăm un model de regresie cu suport vectorial
model_svr = SVR(kernel='linear')
model_svr.fit(X_train.select_dtypes(include=[np.number]), y_train)

# Predicții pe setul de test
y_pred_lr = model_lr.predict(X_test.select_dtypes(include=[np.number]))
y_pred_rf = model_rf.predict(X_test.select_dtypes(include=[np.number]))
y_pred_svr = model_svr.predict(X_test.select_dtypes(include=[np.number]))

# Evaluăm modelele
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Afișăm rezultatele
print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}, R²: {r2_lr:.2f}")
print(f"Random Forest - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.2f}")
print(f"Support Vector Regression - RMSE: {rmse_svr:.2f}, MAE: {mae_svr:.2f}, R²: {r2_svr:.2f}")
# Comparăm performanța modelelor
results = pd.DataFrame({
	'Model': ['Linear Regression', 'Random Forest', 'Support Vector Regression'],
	'RMSE': [rmse_lr, rmse_rf, rmse_svr],
	'MAE': [mae_lr, mae_rf, mae_svr],
	'R²': [r2_lr, r2_rf, r2_svr]
})
print("\nModel Comparison Results:")
print(results)

# Vizualizăm rezultatele cu scatter plots
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

# Visualizăm distribuția erorilor pentru fiecare model
plt.figure(figsize=(12, 6))
sns.histplot(y_test - y_pred_lr, kde=True, color='blue', label='Linear Regression', stat='density')
sns.histplot(y_test - y_pred_rf, kde=True, color='green', label='Random Forest', stat='density')
sns.histplot(y_test - y_pred_svr, kde=True, color='red', label='Support Vector Regression', stat='density')
plt.title('Distribution of Errors')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Density')
plt.legend()
plt.show()
