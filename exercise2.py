import pandas as pd
import matplotlib.pyplot as plt

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

# visualize the relationship between features and target
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['total_bill'], df['tip'], alpha=0.5)
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.subplot(1, 2, 2)
plt.scatter(df['size'], df['tip'], alpha=0.5)
plt.title('Size vs Tip')
plt.xlabel('Size')
plt.ylabel('Tip')
plt.tight_layout()
plt.show()



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

