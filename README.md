# CST2208_Project
ICR - Identifying Age-Related Conditions, use Machine Learning to detect conditions with measurements of anonymous characteristics 
Project Summary 

**Background** 

The project considers the multitude of health issues associated with aging, ranging from heart disease and dementia to hearing loss and arthritis. Aging is a significant risk factor for numerous diseases and complications, and bioinformatics research focuses on interventions that can slow down and reverse biological aging, as well as prevent age-related ailments. Data science plays a crucial role in developing new methods to address these complex problems, even when the available sample size is small. 

**Problem Statement**

In the context of predicting medical conditions, the current use of models like XGBoost and Random Forest may yield unsatisfactory performance, particularly in critical situations where accurate and timely predictions are essential to safeguard lives. Existing methods require improvement to ensure reliable and consistent predictions across diverse cases, especially when dealing with life-critical scenarios. 

This project presents an opportunity to advance the field of bioinformatics by addressing the limitations of current predictive models and exploring innovative approaches to solving critical medical problems. By developing more efficient and accurate prediction methods, the project aims to pave the way for enhanced healthcare interventions and interventions that can potentially mitigate the impact of age-related ailments. The successful implementation of this project has the potential to revolutionize medical decision-making and improve patient outcomes, making it a significant and promising endeavor in the field of predictive analytics and bioinformatics. 

**Objective** 

1. The primary objective of this project is to develop an efficient and reliable machine learning model for identifying age-related conditions based on anonymous health characteristics. The specific goals of this project are as follows: 

2. Predictive Model Development: Build and train a predictive model capable of accurately classifying individuals into two classes: those with age-related medical conditions (Class 1) and those without any age-related conditions (Class 0). 

3. Improved Medical Condition Detection: Improve upon existing methods, such as XGBoost and Random Forest, to enhance the accuracy and robustness of predictions. The aim is to create a model that outperforms these traditional algorithms, especially in critical medical scenarios where timely and accurate detection is essential. 

4. Privacy Preservation: Ensure the privacy of patients by utilizing anonymized health characteristics in the model. The use of key characteristics will enable the encoding of relevant patient details while protecting sensitive information, thus complying with data privacy regulations. 

5. Advancement in Bioinformatics: Contribute to the field of bioinformatics by introducing innovative approaches for addressing age-related ailments. The project seeks to leverage data science and machine learning techniques to identify potential interventions that can slow down or reverse the effects of biological aging. 

6. Sample Size Considerations: Address the challenges posed by limited sample sizes in bioinformatics research. Develop methods and techniques that can yield reliable predictions even when working with a relatively small training dataset. 

 

**Methodology** 

The methodology encompasses data preprocessing, model building, and evaluation stages: 

1. Exploratory Data Analysis (EDA):  

Load the dataset into Pandas DataFrames. 

Perform visualizations and statistical analysis to understand the relationships between the health characteristics and the target variable (Class). 

Address outliers: Apply outlier detection and treatment techniques to enhance the quality of the data. 

Explore and analyze the distributions of features to gain insights into the data. 

Investigate the supplemental metadata (greeks.csv) to identify any patterns or correlations with the target variable. 

 

2. Data Preprocessing: 

Dataset Split with Stratified Sampling 

Transformation 

Scaling 

Imputation 

 

3. Model Building and Training: 

Split the pre-processed data into training and testing sets. 

Implement multiple machine learning models based on the project's objectives, including: 

Logistic Regression 

K-Nearest Neighbors (KNN) 

Support Vector Machines (SVM) with linear and non-linear kernels 

XGBoost 

Decision Trees 

Random Forest 

Naive Bayes 

Catboost 

MLPC 

Train each model using the training data and tune hyperparameters to optimize performance. 

 

4. Model Evaluation: 

As a requirement of the Kaggle Competition, a function for Balanced Logg Loss were used to evaluate the performance of each model. 

Also, to better understand he weakness and strength of each model, other appropriate metrics were used such as accuracy, precision, recall, F1-score, and ROC-AUC. 

Compare the performance of different models to identify the most effective one for age-related condition detection. 

 

5. Model enhancement (Tuning and Ensembles): 

Hyperparameter Tuning Strategy: Before model training, we will conduct hyperparameter tuning for each base model to optimize their performance. Cross-validation techniques will be employed to find the best hyperparameters for each model. The tuning process will be based on relevant evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. 

Addressing Overfitting: We will closely monitor the performance of tuned models on both the training and validation sets. If overfitting is observed, regularization techniques and early stopping will be applied during training to prevent overfitting and improve generalization. 

Ensemble Method: To address any remaining overfitting concerns and improve the overall performance on class 1 predictions, we will implement an ensemble method, specifically the Voting Classifier. The Voting Classifier will combine the predictions of multiple base classifiers (Random Forest, Cat Boost, and XG Boost) using hard voting to make the final prediction. By leveraging the strengths of each model, the ensemble aims to mitigate overfitting and enhance the predictive power of the final model. 

 

6. Model Deployment: 

A. Cleaning and Feature Engineering for Test Data: 

Before deploying the final model, the test data must undergo the same data cleaning and feature engineering steps as performed during the training phase. 

Any missing values in the test data are imputed using the same strategy as applied to the training data. 

Categorical variables are encoded using the same encodings as one-hot encoding as used during training. 

Feature scaling is performed on numerical features to ensure consistency with the training data. 

B. Loading the Trained Model: 

The trained ensemble model "stacking_clf" is loaded into the deployment environment from a serialized file saved during model development. 

Model Prediction with Stacking Classifier: 

The cleaned and preprocessed test data is fed into the "stacking_clf" model for prediction. 

The model generates predictions for the target variable based on the input features. 

C. Handling Predictions: 

The predictions generated by the model can be further post-processed or transformed as required, depending on the specific use case or business needs. 

For binary classification, the model typically outputs probabilities for class 0 and class 1. 

A threshold can be applied to convert probabilities to binary predictions if necessary. 

D. Result Output: 

The final predictions can be stored in a suitable format for further analysis or used for decision-making in the production environment. 

If required, the results can be integrated with other systems, applications, or databases for further processing or visualization. 

 

**Data Evaluation Summary**

The dataset provided for this project comprises two main components: the training dataset and the greeks dataset. 

Training Dataset: The training dataset consists of 617 observations, each containing a unique ID and 56 health characteristics that have been anonymized. These characteristics include 55 numerical features and 1 categorical feature. Alongside the health characteristics, the dataset also includes a binary target variable called "Class." The primary goal of this project is to predict the Class of each observation based on its respective features. 

Greeks Dataset: In addition to the training dataset, there is a supplementary metadata dataset called "greeks." This dataset provides additional information about each observation in the training dataset and encompasses five distinct features. 

By utilizing these datasets, we aim to develop a predictive model that can effectively identify age-related conditions. 

For a more comprehensive understanding of the datasets and to explore the detailed analysis, kindly refer to the accompanying Jupyter notebook. 

 

**Data Dictionary**

train.csv- The training set. Id Unique identifier for each observation. AB-GL 56 anonymized health characteristics. All are numeric except for EJ, which is categorical. Class A binary target: 1 indicates the subject has been diagnosed with one of the three conditions, 0 indicates they have not. 

test.csv - The test set. our goal is to predict the probability that a subject in this set belongs to each of the two classes. 

greeks.csv - Supplemental metadata, only available for the training set. 

Alpha Identifies the type of age-related condition, if present. 

A No age-related condition. Corresponds to class 0. 

B, D, G The three age-related conditions. Correspond to class 1. 

Beta, Gamma, Delta Three experimental characteristics. 

Epsilon The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected. 

 

**Libraries** 

Pandas: For data manipulation and analysis. 

NumPy: For numerical computations and array operations. 

Matplotlib: For data visualization. 

Seaborn: For enhanced data visualization. 

Scikit-learn: For machine learning algorithms and tools. 

XGBoost: For implementing the XGBoost model. 

 

**Step to Follow**

1. EDA and Cleaning: we understand the data as much as we can and do some initial data preparation. 

2. Data preprocessing including dataset split, oversampling, features reduction, transformation, scaling. 

3. Modelling including simple and advanced model with default parameters. 

4. Tunning the parameters of selected models. 

5. Evaluation metrics with best parameters of best models. 

6. Ensemble models to get the best results. 

7. Prediction for submission in Kaggle competition. 
