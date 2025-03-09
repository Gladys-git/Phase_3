# Machine Learning Model Development

## Project Overview/Business Understanding
  - This project focuses on building and evaluating multiple machine learning models to achieve high accuracy in predictions. The goal is to preprocess data, perform feature engineering, train and evaluate models, and select the best-performing one for deployment.
  -   Syria Tel is encountering a challenge with the number of customers that are quitting from their services. Losing customers can be expensive for business due to the lost revenue and expenses that are incurred in acquiring new new customers.
# Problem Statement
  - The challenge is to develop a machine learning model that can perfectly predict whether a customer will churn based in historical usage and demographic data. Identifying patterns and risk factors associated with the moving customers will assist the company implement the targeted retention strategies. The project aims to:
  - Classification Task: Predict whether a customer will leave SyriaTel or remain a subscriber.
  -Pattern Analysis: Identify key factors influencing customer churn to support business decision-making.
# Objectives
  -Develop a predictive model to classify customers as either churners or non-churners.
  -Engineer relevant features from customer usage patterns and demographics.

## Data Understanding
    - The dataset has 5rows and 21 columns. 
## Dataset
- **File Used:** `bigml.data.csv`
- **Description:** Contains categorical and numerical features used to predict a target variable.
- **Preprocessing Steps:**
  - Handling missing values : There were no missing values in the dataset. Dropped columns such as 'Phone Number' as they were not useful for my analysis.
  - Encoding categorical variables : The categorical variables converted to 1/0 from true or false (International Plan, Voice mail Plan, churn).
  # Feature scaling and transformation
  - Checked for outliers (Selected only the numeric columns to preserve the categorical variables).
 - Below is an image that visualizes outliers after cleaning
 ![image](https://github.com/user-attachments/assets/6bdf7d94-21be-4ba4-8759-0bda17ff7ad7)

## Feature Engineering
- Created new features
- Binned certain features for better performance
- Encoded categorical features using frequency encoding
- Correlation Matrix
  ![image](https://github.com/user-attachments/assets/45237a3c-5c42-463f-b0b5-7cc23ff49245)

 # Feature Selection
    - Scaling the X_train and the X_test 
    - Split data into features and target
    - Split into train and test sets
## Modeling
The following models were tested:
1. **Random Forest using Tree-Based Model**. Output = RandomForestClassifier(random_state=42)
2. ****Logistic Regression**. Output = LogisticRegression(max_iter=500)
3. ****XGBoost Model**. Output =
4.                    XGBoost Accuracy: 0.9805
               precision    recall  f1-score   support

       False       0.98      1.00      0.99       566
        True       1.00      0.87      0.93       101

     accuracy                           0.98       667
     macro avg       0.99      0.94      0.96       667
  weighted avg       0.98      0.98      0.98       667
****Decision Tree**. Output = Decision Tree Accuracy: 0.9535
****K-Nearest Neighbors (KNN)**. Output =
          KNN Accuracy: 0.8981

     Model Results Summary:
      Decision Tree: 0.9535
        KNN: 0.8981

## Model Evaluation
- **Metrics Used:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC Curve
- ![image](https://github.com/user-attachments/assets/fd9ca87a-7586-4167-9319-eea8b84243f2)
![image](https://github.com/user-attachments/assets/2d8aa72d-b826-4af8-a612-18131d49a17e)
![image](https://github.com/user-attachments/assets/5b703bbc-a30a-499f-9e80-a9d53924ac99)

- **Best Model:** 🏆 **XGBoost Classifier**  
  - Optimized Accuracy: **0.9805**
  - Best Hyperparameters:  
    ```python
    {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.9}
    ```

## Conclusions & Recommendations
- **XGBoost outperformed** other models in accuracy and recall.
- Feature engineering and proper encoding significantly improved model performance.
- Future improvements: **Deep learning techniques** or further hyperparameter tuning.

