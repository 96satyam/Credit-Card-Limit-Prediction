 ## Problem Statement
Financial institutions need to allocate appropriate credit card limits to customers.
Incorrect allocations can either discourage usage (if too low) or increase credit risk (if too high).

The goal is to build a machine learning model that predicts a suitable credit card limit using customer profile data, and deploy it as a web application for real-time usage.

## Project Workflow
### 1. Exploratory Data Analysis (EDA)
Performed in eda.py

Checked missing values, outlier detection, distribution plots.

Studied feature correlation and importance.

Detected patterns between Income, Age, and Credit Card Limit.

### 2. Data Preprocessing
Handled in data_ingeston.py

StandardScaler used for feature scaling.

Converted categorical variables into numerical.

Outlier treatment for skewed columns.

Train-Test split (80:20) applied.

### 3. Model Building
Multiple models tried:

Linear Regression

Decision Tree Regressor

Random Forest Regressor
XGBoost Regressor

AdaBoost Regressor


AdaBoost Regressor selected based on best performance.

### 4. Modular Coding
Functions are organized into separate Python files:

eda.py for data exploration

model.py for training and evaluation

predict_pipeline.py for making real-time predictions from user input

app.py for Flask web app backend

This makes the codebase clean, scalable, and maintainable.

### 5. Deployment (Flask App)
Created a web app using Flask (app.py).

Frontend designed in simple HTML (templates/index.html).

User inputs are collected through a web form.

Prediction is displayed immediately on submission.



## Key Highlights
#### Full machine learning pipeline: EDA → Preprocessing → Modeling → Evaluation → Deployment

#### Modular coding for easy maintenance

#### Real-time prediction web app using Flask

#### User-friendly frontend for input collection

#### Best practices followed in structure and documentation
