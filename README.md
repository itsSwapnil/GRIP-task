# Predict Student Scores Based on Study Hours

## Project Overview
This project is a part of **The Spark Foundation GRIP Task 1** for the **Data Science and Business Analyst Internship**. The objective is to build a simple linear regression model to predict the percentage of marks a student is likely to score based on the number of study hours.

---

## Problem Statement
Predict the percentage of a student’s score based on the number of hours they study.

---

## Dataset
The dataset used contains two columns:
- **Hours:** Number of study hours
- **Scores:** Percentage of marks scored

---

## Libraries Used
The following Python libraries are required:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

---

## Project Workflow

### 1. Data Import and Exploration
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
students = pd.read_excel("path_to_dataset")
print(students.head())
print(students.describe())
print(students.isna().sum())
```

### 2. Data Visualization
#### Scatter Plot
```python
plt.scatter(students.Hours, students.Scores, color="c")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
#### Heatmap for Correlation
```python
correlation = students.corr()
sns.heatmap(correlation, annot=True)
plt.show()
```
#### Boxplot for Outlier Detection
```python
fig = plt.figure(1, figsize=(12, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(students, patch_artist=True)
```

### 3. Data Splitting
```python
X = students[['Hours']]
y = students[['Scores']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

### 4. Model Training
```python
linear = LinearRegression()
model = linear.fit(x_train, y_train)
```

### 5. Model Evaluation
```python
y_pred = model.predict(x_test)
plt.scatter(x_train, y_train, color='black')
sns.regplot(x_train, linear.predict(x_train), color='r')
plt.title("Hours Studied Vs Scores Obtained")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.grid(True, color='white', linestyle='-.')
plt.show()
```

### 6. Predictions
```python
h = int(input("Enter study hours: "))
predictions = model.predict([[h]])
print(f"If a student studies {h} hours daily, then the predicted score is {predictions[0]}")

pred2 = model.predict([[9.25]])
print(f"If a student studies 9.25 hours/day, the predicted score is {pred2[0]}")
```

### 7. Model Performance Metrics
```python
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
MSE = format(mean_squared_error(y_test, y_pred), '.3f')
r2 = format(r2_score(y_test, y_pred), '.4f')

print(f"Root Mean Squared Error: {RMSE}")
print(f"Mean Squared Error: {MSE}")
print(f"R2 Score: {r2}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
```

### Final Model Accuracy
The model provides **95.68% accuracy**, making it a reliable predictor for the task.

---

## Conclusion
This project demonstrates the use of simple linear regression to solve a real-world problem of predicting student scores based on their study hours. With proper data preprocessing, visualization, and model evaluation, the accuracy of the predictions reached a high level.

---

## How to Run
1. Clone this repository.
2. Install the required dependencies using:
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn
   ```
3. Run the notebook or Python script.
4. Input study hours to see the predicted scores.

---

## License
This project is licensed under the MIT License.

