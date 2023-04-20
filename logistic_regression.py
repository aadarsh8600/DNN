import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Drop columns that we won't use in the model
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill in missing values for the 'Age' and 'Embarked' columns
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numerical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'])

# Split the dataset into training and test sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the appropriate logistic regression model
model = LogisticRegression(penalty='l2', random_state=42)

# Train the logistic regression model
model.fit(X_train, y_train)

# Evaluate the performance of the logistic regression model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Load the test dataset
test_df = pd.read_csv('test.csv')

# Preprocess the test dataset
test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df = pd.get_dummies(test_df, columns=['Embarked'])

# Make predictions on the test dataset
test_predictions = model.predict(test_df)

# Save the predictions to a CSV file for submission to Kaggle
submission_df = pd.DataFrame({'PassengerId': pd.read_csv('test.csv')['PassengerId'], 'Survived': test_predictions})
submission_df.to_csv('submission.csv', index=False)
