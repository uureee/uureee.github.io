---
title: "Predicting Titanic Survival Using Logistic Regression"
layout: post
post-image: "https://raw.githubusercontent.com/thedevslot/WhatATheme/master/assets/images/What%20is%20Jekyll%20Banner.png"
description: "A beginner-friendly guide to predicting Titanic passenger survival using Python and logistic regression."
tags:
- machine-learning
- python
- data-science
- tutorial
---

In this post, we’ll use Python to predict whether a passenger on the Titanic survived using **logistic regression**.  
We’ll go through the data cleaning, training, and testing steps — explained in **simple beginner-friendly language**.

---

## 1. Importing Libraries

We start by importing the main Python libraries we’ll use.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```

These libraries help us load data, visualize patterns, and build a machine learning model.

---

## 2. Loading the Dataset

We load the famous **Titanic dataset**.

```python
data = pd.read_csv('train.csv')
data.head()
```

This dataset contains information about Titanic passengers, including their class, age, gender, and whether they survived.

---

## 3. Checking Missing Data

```python
data.isnull().sum()
```

We find missing values in the `Age`, `Cabin`, and `Embarked` columns.

---

## 4. Handling Missing Values

We fill missing **Age** values with the **average age** of passengers having the same gender and class.

```python
data['Age'] = data.groupby(['Sex', 'Pclass'], group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))
```

This keeps the average age realistic for each group.

---

## 5. Why We Don’t Use “Survived” for Filling Age

We avoid using the **Survived** column when filling in missing ages because it would cause **data leakage** — giving the model unfair information from the target variable.

---

## 6. Converting Text Columns to Numbers

Machine learning models work only with numbers, so we convert text columns into numeric ones.

```python
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
```

This creates columns like `Sex_male`, `Embarked_Q`, and `Embarked_S`.

---

## 7. Splitting Features and Target

We define:
- **Features (X):** columns we use for prediction.
- **Target (y):** the column we want to predict.

```python
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']
```

---

## 8. Train-Test Split

We split the dataset into training and testing parts so we can test the model later.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 9. Training the Logistic Regression Model

```python
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

The model learns from the training data.

---

## 10. Making Predictions

```python
y_pred = model.predict(X_test)
```

This predicts survival (1) or not (0) for the test data.

---

## 11. Evaluating the Model

We measure the accuracy and check the confusion matrix.

```python
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
```

Example output:

```
Accuracy: 0.85
Confusion Matrix:
[[46  8]
 [ 5 31]]
```

This means our model predicted correctly about 85% of the time.

---

## 12. Checking Feature Importance

We can see which features were most important in predictions.

```python
feature_importance = model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
```

This shows which features (like **Sex** or **Pclass**) most affected survival chances.

---

## 13. Adding a New Feature: Family Size

Let’s create a new feature that combines siblings and parents onboard.

```python
data['FamilySize'] = data['SibSp'] + data['Parch']
```

We can then include it in our next model.

---

## 14. What We Learned

- How to clean missing data  
- Why not to use the target variable for imputation  
- How to convert categorical data  
- How to train and evaluate a logistic regression model  
- How to identify important features  

---

### 🧠 Key Takeaway
Even a simple model like logistic regression can teach us a lot about data preparation and feature importance.  
Start small, experiment, and learn as you go — that’s the best way to grow as a data scientist!
