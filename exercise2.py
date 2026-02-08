import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

df = pd.read_csv(url)
print(df)


feature =df[["total_bill", 'size']]
target = df['tip']

print("Features: \n", feature.columns.tolist())
print("Target: \n", target.name)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)



print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)



# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# from sklearn.metrics import mean_squared_error, r2_score
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)

