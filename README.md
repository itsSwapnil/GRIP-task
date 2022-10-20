# GRIP-task
# The Spark Foundation - GRIP task 1
## Data Science and Business Analyst Intern
Problem statement : Predict the percentage of students based on the number of study hours.

# Import Libraries and Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')

students = pd.read_excel(r"C:\Users\Swapnil\OneDrive\Desktop\Machine Learning\datasets\gripStudents.xlsx")
students.head()
students.describe()
students.isna().sum()

plt.scatter(students.Hours, students.Scores, color="c")
#plt.plot(height, weight, color="c")

plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()

correlation = students.corr()
sns.heatmap(correlation, annot = True)

fig = plt.figure(1, figsize=(12,7))

### Finding and treating outliers
ax = fig.add_subplot(111)
bp = ax.boxplot(students, patch_artist=True)

### Splitting the data for Training and Testing
X = students[['Hours']]
y = students[['Scores']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

linear = LinearRegression()
model = linear.fit(x_train, y_train)
model
y_pred = model.predict(x_test)

plt.scatter(x_train, y_train, color='black')

sns.regplot(x_train, linear.predict(x_train), color='r')

plt.title("Hour studied Vs Scores obtained")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.grid(True,color='white', linestyle='-.')

y_pred

h = int(input())
predictions = model.predict([[h]])
print("If a student studying",h,"Hrs daily, then score will be",predictions[0])

pred2 = model.predict([[9.25]])
print("If students studies 9.25 hrs/day then predicted score of that students will be", pred2)

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f'))
MSE = format(mean_squared_error(y_test, y_pred), '.3f')
r2 = format(r2_score(y_test, y_pred),'.4f')

print('Root Mean Squared Error =',RMSE, '\nMean Squared Error =',MSE,'\nR2 Score =', r2)

mean_absolute_error(y_test, y_pred)

### This model gives 95.68 % of accuracy
