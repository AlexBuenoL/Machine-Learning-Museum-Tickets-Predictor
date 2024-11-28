"""
    In this python script, some regression models are trained in order to predict 
    the number of museum tickets that will be sold in the next week. 
"""

__author__ = 'Alex Bueno'


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, HuberRegressor
from apafib import load_BCN_museos


# First part: load dataset and data visualization

df = load_BCN_museos()
df.head()

X = df.loc[:, df.columns != 'EntradasMuseos-Museos+1']
y = df['EntradasMuseos-Museos+1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)
train_set = pd.concat([X_train, y_train], axis=1)

# Check data types and null values
print(f'Número de columnas: {len(train_set.columns.to_list())}\n')
print(train_set.dtypes)
print(train_set.isna().sum())

# Show correlation matrix
sns.heatmap(train_set.corr(), cmap='seismic',  center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.figure(figsize=(10, 8))
plt.show()

# Check variables distributions
data_columns = ['EntradasMuseos-Museos+1', 'AeronavesDestino-Europa', 'VisitantesPais-Regne Unit',
    'EscalasTipoBarco-Creuers i Iots', 'ComprasSectores-Béns i serveis de primera necessitat']
sns.pairplot(train_set[data_columns])
plt.show()


# Second part: compare Linear, Ridge and Lasso regression

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_train)

r2 = r2_score(y_train, y_pred_lr)
cv_r2 = np.mean(cross_val_score(lr, X_train, y_train))
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_train, y_pred_lr)

print(f'Linear R²: {r2}')
print(f'Linear CV R²: {cv_r2}')
print(f'Linear MSE: {mse}')
print(f'Linear MAE: {mae}')


lambdas = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 5, 10, 50, 100]

# Ridge regression
ridge = RidgeCV(alphas=lambdas, cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_train)

r2 = r2_score(y_train, y_pred_ridge)
cv_r2 = ridge.best_score_
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_train, y_pred_ridge)

print(f'Ridge R²: {r2}')
print(f'Ridge CV R²: {cv_r2}')
print(f'Ridge MSE: {mse}')
print(f'Ridge MAE: {mae}')


# Lasso regression
lasso = LassoCV(alphas=lambdas, cv=5, max_iter=20000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_train)

r2 = r2_score(y_train, y_pred_lasso)
cv_r2 = np.mean(cross_val_score(lasso, X_train, y_train))
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_train, y_pred_lasso)

print(f'Lasso R²: {r2}')
print(f'Lasso CV R²: {cv_r2}')
print(f'Lasso MSE: {mse}')
print(f'Lasso MAE: {mae}')


# Predictions with test data

y_pred_lr = lr.predict(X_test)

r2 = r2_score(y_test, y_pred_lr)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test, y_pred_lr)

print(f'Linear Regression R²: {r2}')
print(f'Linear Regression MSE: {mse}')
print(f'Linear Regression MAE: {mae}')


y_pred_ridge = ridge.predict(X_test)

r2 = r2_score(y_test, y_pred_ridge)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test, y_pred_ridge)

print(f'Ridge R²: {r2}')
print(f'Ridge MSE: {mse}')
print(f'Ridge MAE: {mae}')


y_pred_lasso = lasso.predict(X_test)

r2 = r2_score(y_test, y_pred_lasso)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test, y_pred_lasso)

print(f'Lasso R²: {r2}')
print(f'Lasso MSE: {mse}')
print(f'Lasso MAE: {mae}')


# Show regression plots
def show_real_vs_prediction(y_pred, y_test, n, title):
    plt.subplot(1, 3, n)  
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(title)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')

plt.figure(figsize=(18, 6))

show_real_vs_prediction(y_pred_lr, y_test, 1, 'Regresión Lineal')
show_real_vs_prediction(y_pred_ridge, y_test, 2, 'Ridge')
show_real_vs_prediction(y_pred_lasso, y_test, 3, 'Lasso')

plt.tight_layout()
plt.show()


def show_res(y_pred, y_test, n, title):
    residuos = y_test - y_pred
    plt.subplot(1,3,n)
    plt.scatter(y_pred, residuos, color='blue', alpha=0.6)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.title('Gráfico de Residuos ' + title)
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    
plt.figure(figsize=(20,6))

show_res(y_pred_lr, y_test, 1, 'Regresión Lineal')
show_res(y_pred_ridge, y_test, 2, 'Ridge')
show_res(y_pred_lasso, y_test, 3, 'Lasso')

plt.tight_layout()
plt.show()


# Third part: apply log transformation to the predict variable
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Re-train the models
lr_log = LinearRegression()
lr_log.fit(X_train, y_train_log)

ridge_log = RidgeCV(alphas=lambdas, cv=5)
ridge_log.fit(X_train, y_train_log)

lasso_log = LassoCV(alphas=lambdas, cv=5, max_iter=20000)
lasso_log.fit(X_train, y_train_log)

# Evaluate train predictions
y_pred_lr_log = lr_log.predict(X_train)
r2 = r2_score(y_train_log, y_pred_lr_log)
cv_r2 = np.mean(cross_val_score(lr_log, X_train, y_train_log))
print(f'Lineal: R²={r2}, CV R²={cv_r2}')

y_pred_ridge_log = ridge_log.predict(X_train)
r2 = r2_score(y_train_log, y_pred_ridge_log)
cv_r2 = ridge_log.best_score_
print(f'Ridge:  R²={r2}, CV R²={cv_r2}')

y_pred_lasso_log = lasso_log.predict(X_train)
r2 = r2_score(y_train_log, y_pred_lasso_log)
cv_r2 = np.mean(cross_val_score(lasso_log, X_train, y_train_log))
print(f'Lasso:  R²={r2}, CV R²={cv_r2}')


# Evaluate test predictions
y_pred_lr_log = lr_log.predict(X_test)
r2 = r2_score(y_test_log, y_pred_lr_log)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test_log, y_pred_lr_log)
print(f'Lineal: R²={r2}, MSE={mse}, MAE={mae}')

y_pred_ridge_log = ridge_log.predict(X_test)
r2 = r2_score(y_test_log, y_pred_ridge_log)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test_log, y_pred_ridge_log)
print(f'Ridge: R²={r2}, MSE={mse}, MAE={mae}')

y_pred_lasso_log = lasso_log.predict(X_test)
r2 = r2_score(y_test_log, y_pred_lasso_log)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test_log, y_pred_lasso_log)
print(f'Lasso: R²={r2}, MSE={mse}, MAE={mae}')


plt.figure(figsize=(18, 6))

show_real_vs_prediction(y_pred_lr_log, y_test_log, 1, 'Regresión Lineal')
show_real_vs_prediction(y_pred_ridge_log, y_test_log, 2, 'Ridge')
show_real_vs_prediction(y_pred_lasso_log, y_test_log, 3, 'Lasso')

plt.tight_layout()
plt.show()


plt.figure(figsize=(20,6))

show_res(y_pred_lr_log, y_test_log, 1, 'Regresión Lineal')
show_res(y_pred_ridge_log, y_test_log, 2, 'Ridge')
show_res(y_pred_lasso_log, y_test_log, 3, 'Lasso')

plt.tight_layout()
plt.show()


# Last part: Huber regressor

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

huber = HuberRegressor(max_iter=10000)

param_grid = {
    'epsilon': [1.1, 1.2, 1.35, 1.5],  
    'alpha': [0.0001, 0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(huber, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_sc, y_train_log)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_huber = best_model.predict(X_train_sc)
r2 = r2_score(y_train_log, y_pred_huber)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_train_log, y_pred_huber)
print(f'R²={r2}, MSE={mse}, MAE={mae}')

y_pred_huber = best_model.predict(X_test_sc)
r2 = r2_score(y_test_log, y_pred_huber)
mse = 1-r2 # MSE normalizada
mae = mean_absolute_error(y_test_log, y_pred_huber)
print(f'R²={r2}, MSE={mse}, MAE={mae}')


plt.figure(figsize=(10,5))

# Valores reales vs predicciones
plt.subplot(1, 2, 1)
plt.scatter(y_test_log, y_pred_huber, alpha=0.6)
plt.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'k--', lw=2)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')

# Residuos
residuos = y_test_log - y_pred_huber
plt.subplot(1,2,2)
plt.scatter(y_pred_huber, residuos, color='blue', alpha=0.6)
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.xlabel('Predicciones')
plt.ylabel('Residuos')

plt.tight_layout()
plt.show()




