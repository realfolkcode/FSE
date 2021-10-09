from useful_package.module_a_dev import polynom_3
from useful_package.module_b_dev import hyperbola
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

X = np.linspace(1, 10, 100, endpoint=False)
model = RandomForestRegressor()

y_true = polynom_3(X)
model.fit(X, y_true)
y_pred = model.predict(y_true)
print('MSE polynom:', mean_squared_error(y_true, y_pred))

y_true = hyperbola_3(X)
model.fit(X, y_true)
y_pred = model.predict(y_true)
print('MSE hyperbola:', mean_squared_error(y_true, y_pred))

