import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor

# using a standard scaler for standard scaler
from sklearn.preprocessing import StandardScaler

# also using pipeline to chain multiple steps together
from sklearn.pipeline import Pipeline

# tie it together! find the proper parameters for this model
from sklearn.model_selection import GridSearchCV

print(load_diabetes(return_X_y=True))

x, y = load_diabetes(return_X_y=True)

# same format for each model these two are the same!
# model = LinearRegression()
model = KNeighborsRegressor()
model.fit(x, y)

# creating this new 'pipeline' object
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
])

# make sure to print this to find the exact parameters that you need
pipe.get_params()

better_model = GridSearchCV(estimator=pipe,
             param_grid={'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
             cv=3)

better_model.fit(x, y)

# we also get a very interesting dataframe result from using the GridSearch property. Let's show using pandas!
df = pd.DataFrame(better_model.cv_results_)




# run prediction AFTER fitting the model
pred = better_model.predict(x)

# let's see the results on the predicted values compared to the actual values (y)
plt.scatter(pred, y)

# show the results and everything looks pretty good! everything sh ows as intended... ie. higher x = higher y is good!
plt.show()

# but what happens if you need to preprocess data? I.e. the data has an x that goes to the thousands but the y doesn't
# after preprocessing with pipe, we can clearly see the prediction is more accurate.