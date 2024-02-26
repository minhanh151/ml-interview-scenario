# Predicting Loan Default Using Machine Learning 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 


In the banking and finance sector, lending decisions are pivotal for sustaining operations and meeting market demand. Across various banks, the allocation of loans to diverse customers varies, contingent upon several critical factors pivotal for loan approval or rejection. However, within this framework, there exists a looming concern – the possibility of loan default, where a borrower fails to repay the borrowed sum, poses a substantial risk to financial institutions.  It becomes important for banks to understand the behavior of the __customer__ before they can take action to lend money to people for different purposes. Recognizing the intricate interplay of factors influencing loan repayment behavior, banks are increasingly turning to machine learning techniques for predictive analytics to mitigate the risk of loan default.


## Machine Learning Analysis 

By leveraging machine learning algorithms, organizations can analyze key features and predict outcomes, aiding in the determination of loan eligibility for individuals. The ability to accurately predict whether a customer is likely to default on a loan based on a set of features is invaluable. Machine learning offers a viable solution to address this challenge, empowering businesses to enhance risk assessment and optimize lending practices. In this project, we present the process of building a robust ML model predicting loan default likelihood, incorporating various data sources to improve loan approval processes and reduce default rates.


### I. Data Integration and Privacy Compliance
Ensure compliance with data protection laws (e.g., GDPR, CCPA) and integrate disparate data sources into a unified dataset.

### II. Cleaning and Preprocessing Data
Conduct EDA to understand data distributions, correlations, and patterns. Clean and preprocess data, addressing missing values, outliers, and feature encoding.

### III. Exploratory Data Analysis (EDA)
Conduct EDA to understand data distributions, correlations, and patterns. Clean and preprocess data, addressing missing values, outliers, and feature encoding.

### IV. Feature Engineering
Develop features capturing customer risk profiles. Utilize domain knowledge for predictive feature construction.

### MV. Model Development and Selection
Experiment with various ML models and perform hyperparameter tuning for optimization.

### VI. Model Evaluation
Use metrics like AUC-ROC, accuracy, F1-score, and precision-recall. Focus on minimizing false negatives impacting financial decisions.

### VII. Hyperparameter Tuning
Hyperparameter tuning allows data scientists to tweak model performance for optimal results.

### VIII. Bias and Fairness Assessment
Evaluate and mitigate biases to ensure fair lending practices.

### IX. Deployment and Monitoring
Develop a deployment strategy and establish monitoring for performance and data drift.

### X. Feedback Loop and Continuous Improvement
Set up a system for insights from loan officers and customers. Regularly update the model with new data.

## Deliverables
- **Technical Report.pdf** - A high-level and concise report on the approach, architecture decisions, development process, and model insights.
- **README.md** - A comprehensive guideline and analysis report detailing the approach, architecture decisions, development process, and model insights.
- **Loan_Default_Prediction.ipynb** - A predictive model for identifying high-risk loan applicants.
- **Deployment Strategy.pdf** A deployment strategy with monitoring and updating protocols.

**NOTE: After some analysis of the provided data, we can see that this data is synthetically duplicated with the same values which cannot be used to train Machine Learning models for fraud detection. So we will suggest some insights and highlight techniques based on the common knowledge and state-of-the-art Machine Learning models for loan default prediction.**

## Feature Engineering

Feature engineering plays a crucial role in loan default prediction, as it involves selecting, creating, and transforming features from raw data to improve the predictive performance of machine learning models. 

From the provided bank transactional data, we can can derive various attributes that may be indicative of a customer's financial behavior and potential default risk. Here are some example features we can create:

1. Transaction Amount Statistics:

- Mean transaction amount over the last 7 - 30 - 60 days

- Median transaction amount over the last 7 - 30 - 60 days

- Standard deviation of transaction amount over the last 7 - 30 - 60 days

- Total transaction amount over the last 7 - 30 - 60 days

2. Transaction Frequency:

- Number of transactions over the last 7 - 30 - 60 days

- Number of transactions during different time periods (e.g., business hours, weekends)

3. Transaction Categories:

- Number of different transaction categories (e.g., groceries, utilities, entertainment) - bank_transaction_code

- Frequency of transactions in some main category

4. Transaction Patterns:

- Regularity of transactions (e.g., presence of recurring payments) Time between transactions (e.g., average time between transactions)

- Time since the last transaction
  
5. Account Balance Trends:

- Average account balance

- Minimum and maximum account balance

- Account balance volatility (e.g., standard deviation of account balance)

6. Distribution of transactions across different locations (e.g., city, country)

- Frequency of transactions in each location
  
7. Payment Behavior:

- Proportion of successful payments

- Proportion of failed payments


## Sampling Methods

Class imbalance is one of the most difficult challenges for machine learning models in loan default prediction. Class imbalance refers to the scenario where one class (e.g., loan defaults) is significantly less frequent than another class (e.g., non-defaults). Since we are dealing with data that is not balanced, it is important to perform the balancing, especially for the minority class which in our case is the possibility that a person might default on a loan. We use various sampling techniques such as __SMOTE__ and __Random sampling__ to get the best outputs from the machine learning models. 

* [__RandomOverSampler__](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
* [__TomekLinks__](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html)
* [__SMOTE__](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
* [__NearMiss__](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.NearMiss.html)

## Metrics

The output variable in our case is __discrete__. Therefore, metrics that compute the outcomes for discrete variables should be taken into consideration and the problem should be mapped under classification. Below are the metrics for the classification problem of predicting whether a person would default on a loan or not. 

* [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1 Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__Area Under Curve (AUC)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html)
* [__Classification Report__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
* [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)



## ML Models

Determining the best-performing machine learning method for loan default prediction depends on various factors such as dataset characteristics, problem complexity, and performance metrics. Here are several machine learning methods commonly used in loan default prediction and their respective strengths:

### Naive Bayes:
#### Strengths:
- Naive Bayes is a simple and computationally efficient algorithm, making it easy to implement and understand.
-  Naive Bayes is commonly used in text classification tasks, where the features are typically discrete (e.g., word counts), making it suitable for loan application text analysis.
#### Use Case: 
The dataset contains a large number of discrete features (e.g., categorical or text-based features).
Interpretability and computational efficiency are prioritized over predictive performance.

### Logistic Regression:
#### Strengths:
- Simple and interpretable model, providing insights into feature importance and coefficients.
- Well-suited for binary classification tasks like loan default prediction.
- Efficient computation, making it suitable for large datasets and real-time applications.
#### Use Case: 
Logistic regression is effective when the relationship between features and loan default is predominantly linear, and interpretability is crucial.

### Random Forest:
#### Strengths:
- Robust ensemble learning method that can handle non-linearity, feature interactions, and noisy data.
- Less prone to overfitting compared to individual decision trees.
- Can automatically handle feature selection and provide feature importance rankings.
#### Use Case: 
Random forest is beneficial when dealing with complex datasets with non-linear relationships between features and loan default.

### Gradient Boosting Models (e.g., XGBoost, LightGBM):
#### Strengths:
- Excellent predictive performance, often outperforming other models in terms of accuracy and generalization.
- Handles complex relationships and interactions between features effectively.
- Robust to overfitting through regularization techniques and early stopping.
#### Use Case: 
Gradient boosting models are suitable for high-dimensional datasets with complex patterns and when maximizing predictive accuracy is paramount.


### Metric Evaluations

#### False Positive (Type I Error):
- Definition: A false positive occurs when the model predicts that a loan will default (positive class), but in reality, the borrower does not default on the loan (negative class).

- Impact: False positives can lead to unnecessary financial losses for lenders by rejecting loan applications from creditworthy borrowers. Additionally, false positives can result in customer dissatisfaction and loss of potential business.

#### False Negative (Type II Error): => More dangerous and important type
- Definition: A false negative occurs when the model predicts that a loan will not default (negative class), but in reality, the borrower defaults on the loan (positive class).

- Impact: False negatives are costly for lenders as they result in missed opportunities to identify high-risk borrowers who may default. This can lead to financial losses due to unpaid loans and may also harm the lender's reputation.



### Model Performance

#### Random Oversampling 
In this set of visualizations, let us focus on the model performance on unseen data points. Since this is a __binary classification task__, metrics such as precision, recall, f1-score, and accuracy can be taken into consideration. Various plots that indicate the performance of the model can be plotted such as confusion matrix plots and AUC curves. 
Let us look at how the models usually perform in the test data. 

[__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - This usually is the first model used to make a prediction about the chances of a person defaulting on a loan. Overall, it often does a good job of classifying defaulters. However, there are many false positives and false negatives in this model. This could be mainly due to high bias or lower complexity of the model. 

AUC curves give a good idea of the performance of ML models. When using logistic regression, it is usually seen that the AUC is about 0.50 - 0.56 respectively. This means that there is a lot more room for improvement in performance. The higher the area under the curve, the better the performance of ML models. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/LR%20AUC%20Curves.png"/>

[__Naive Bayes Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - This classifier works well when there is textual information. Usually, there is a large number of false negatives when using Naive Bayes Classifier for loan default prediction. False negatives mean that the model predicted a defaulter as a non-defaulter. As a result, banks might have a higher chance of losing income especially if money is lent to defaulters. Therefore, we can go ahead and look for alternate models. 

The AUC curves also showcase that the model needs improvement. The AUC of the model is around 0.45 - 0.52 respectively. We can also look for alternate models that can improve performance even further. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/NB%20AUC%20Curves.png"/>

[__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - Based on previous research, the performance of the decision tree classifier is usually better than logistic regression and Naive Bayes. However, there are still possibilities for improvement of model performance even further. We can explore another list of models as well. 

Based on the results generated from the AUC curve from a loan default prediction use case, there is an improvement in the score compared to logistic regression and decision tree classifier. However, we can test a list of other possible models to determine the best for deployment. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/DT%20AUC%20Curves.png"/>

[__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - They are a group of decision trees that ensure that there is less variance during training. In some case, however, the model might not perform well on its positive predictions. This can be due to the sampling approach chosen for training the models. In the later parts, we can focus our attention on other sampling methods. 

After looking at the AUC curves, it can be seen that better models and over-sampling methods can be chosen to improve the AUC scores. Let us now do SMOTE oversampling to determine the overall performance of ML models. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/RF%20AUC%20Curves.png"/>

#### SMOTE Oversampling 

[__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - In this analysis, the same decision tree classifier is trained but using SMOTE oversampling method. The performance of the ML model can improve significantly with this method of __oversampling__. Focusing our attention on the AUC curves, in common scenarios, there can be a significant improvement in the performance of the decision tree classifier. The AUC score is about 0.7-0.83. Therefore, SMOTE oversampling can be used as an effective method to improve the overall performance of the classifier. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/DT%20SMOTE%20AUC%20Curves.png"/>

We can also try a more robust model such as a random forest and determine the performance of the classifier. 


[__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - When training random forest model on SMOTE oversampled data, there can be a good improvement in the performance of the models. It is able to accurately predict the chances of default on a loan. The performance of the random forest classifier is exceptional as it is able to give an AUC score of around  __0.85__ to __0.95__. Therefore, we can deploy this model in real-time as it shows a lot of promise in predicting the chances of applicants defaulting on a loan. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/RF%20SMOTE%20AUC%20Curves.png"/>

## Machine Learning Models' Results:

If our dataset contains __millions of records__, it is important to use the most appropriate machine learning model that deals with __high-dimensional data__ well. Below are the results of machine learning models used for predicting whether a person would default on a __loan or not__ in a sample use case. 

| __Machine Learning Models__| __Accuracy__| __Precision__|__Recall__|__F1-Score__| __AUC Score__|
| :-:| :-:| :-:| :-:| :-:| :-:|
| [__1. Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)| 64.5%| 0.64| 0.63| 0.63| 0.69|
| [__2. Naive Bayes Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)| 50.0%| 0.50| __0.99__| 0.70| 0.64|
| [__3. Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)| 81.0%| __0.76__| 0.84| 0.80| 0.81|
| [__4. Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)| __86.0%__| 0.74| 0.98| __0.84__| __0.95__|
| [__5. XGBoost Classifier__](https://xgboost.readthedocs.io/en/stable/)| __86.0%__| 0.73| 0.96| __0.84__| __0.94__|
| [__6. Deep Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)| 73.0%| 0.66| 0.77| 0.71| 0.76|



As the most of the prior research, we could expect that ensemble tree-based methods such as Random Forest and Gradient Boosting Models (e.g., XGBoost, LightGBM) are often considered among the most effective approaches for loan default prediction. Their effectiveness comes from their ability to handle complex data, provide insights into feature importance, and deliver high predictive accuracy, making them valuable tools for lenders in assessing and managing credit risk. 


