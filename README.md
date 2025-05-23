# Classification Machine Learning Project on Loan Default Data
**Technology**: Python, Pandas, NumPy, Scikit-Learn, tensorflow, matplotlib, seaborn
<br>
**Data Science method**: Data Manipulation, Data Aggregation, Data Visualization, Logistic Regression, Decision Tree, Supporting Vector Machine, ANN

## Executive Summary
1.	People at younger age would have less education and lower income and less savings and most of them are singles, so they don’t have much money to earn a high credit score; therefore, they would end up having a loan at a higher interest rate. If they lost their job, they could easily go default.
2.	When originate a loan, the bank ought to focus on a client’s earning power. Normally, a higher earner has a high degree, more working experiences.
3.	Also focus on co-signers. If they are married or have a co-signer with earning powers to share the financial burdens, they could avoid the default at time of crisis.
4. ANN have a great accuracy of 88%, but terrible at predicting positive cases. The precision is 61% and the recall is 2%.  If only focus on accuracy, ANN would be recommended. If positive cases are more valued, then logistic regression would be recommended.

## Case Descriptions: 
The project focuses on a loan default data from a bank. The analysis will extract insights regarding to clients who had default on loans they had taken from the bank. To protect the bank from potential future delinquencies, three machine learning models were built to automate the tasks to flag customers who could potentially default on their loans.



## Dataset Descriptions:
This is a single table dataset. It has 255347 rows and 18 columns. It contains clients loan information and their default status. There is neither null values nor duplicated values. All the features are in the right data types.

![image](https://github.com/user-attachments/assets/02d45c7f-89c8-42a2-b2ea-173edd4d5d98)

![image](https://github.com/user-attachments/assets/06bd1cd4-3568-40f0-b40d-dbea2b409c96)


## Part 1: EDA & Visualization
1.	Clients age ranges from 18 and 69. Loan amount goes from 5000 to 249999. Their working experiences are between 0 years and 10 years. 
2.	Interest rate of those loans are between 2% and 25%, and the loan term are between 12 months and 60 months.

![image](https://github.com/user-attachments/assets/3928533f-66a2-46b7-a420-bc82c55daea1)


3.	Each feature’s distribution is uniform. Each category or each bin has equal amount of data points.
 
![image](https://github.com/user-attachments/assets/34801ea1-9ae4-4796-be62-3d736fe8e89d)

![image](https://github.com/user-attachments/assets/aa6717f6-852e-4c2d-89dc-ba301d49008e)
 
4.	Total number of customers is 255347. There are 29653 customers who had defaulted on their loans, and 225694 customers who didn’t default on their loans. Therefore, the default rate is 11.61%.
5.	The default customers are younger, have less income, lower credit scores, less working months. However, they have higher interest rate on their loans, and their loans tend to be larger. Therefore, their debt-to-income ratios tend to be higher.

![image](https://github.com/user-attachments/assets/3c4d7be3-829a-490e-b867-afe6aee61683)


6.	Correlation Analysis. All the features are not correlated with each other. However, the default feature is negatively correlated with age, income, credit score, and months employed. The default feature is positively correlated with number of credit lines, interest rate, and DTI ratio or debt to income ratio.

![image](https://github.com/user-attachments/assets/e9790b18-71c5-4e4a-81fd-fa0199c111ec)

 
7.	Once again, younger a group of customers is, the higher the group’s loan default rate is.

![image](https://github.com/user-attachments/assets/340c3890-187e-4ae6-96df-5141533b4aad)


8.	People with higher educations, longer working hours, a marriage and a cosigner tend to have a lower default rate.

![image](https://github.com/user-attachments/assets/f2796e4c-7563-43cf-bba7-5e2f48b6833a)


9.	Groups with lower age, lower income, higher loan amount, lower credit score, lower working hours, but higher interest rate on the loans would have a higher default rate.

![image](https://github.com/user-attachments/assets/974bd6db-ef2e-4d12-911e-7edb4cadcff7)

 
10.	All the numerical features tend to have the same distribution for each category of all the categorical features.
 
![image](https://github.com/user-attachments/assets/2f69042f-5829-4de9-bd79-803449a598fe)


## Part 2.1: Logistic Regression
1.	Data pre-processing. One hot encoding with dropping the first encoded category. Scale the data with standard scaler.
2.	Model tuning. Parameters are C of np.linspace(0.1,2,20) and penalty of l1 and l2. The best params are C=0.8 and penalty=l1.
3.	Model evaluation. Training accuracy is 67%, testing accuracy is 67%. No Overfitting. Precision for positive case is 22% and recall is 70%. The model is better at predicting negative cases over positive cases.

![image](https://github.com/user-attachments/assets/eb9785fa-3d07-45bf-a7fa-8dfccac236a5)


## Part 2.2: Decision Tree Classifier
1.	Data pre-processing. One hot encoding without dropping the first encoded category. No data scaling.
2.	Model tuning. The best params are criterion = gini, max depth=20, min sample leaf is 5.
3.	Important features are Income, Age, Months Employed, credit score, interest rate. 

![image](https://github.com/user-attachments/assets/6969ce7c-ece3-4391-80cd-dff628db240e)

 
4.	Model evaluation. Training accuracy is 80%, testing accuracy is 69%. Very Overfit. 

5.	Precision for positive case is 18% and recall is 47%. The model is better at predicting negative cases over positive cases.

![image](https://github.com/user-attachments/assets/1350db53-ddc0-4e84-a0ad-e2b6c33dd35e)

 
## Part 2.3: Artificial Neural Network
4.	Data pre-processing. One hot encoding with dropping the first encoded category. Scale the data with standard scaler.
5.	ANN architect. Input layer 10 neuros. 2 hidden layers with 10 neuros each. Output layer 1 neuro.
6.	Model evaluation. Training accuracy is 88%, testing accuracy is 88%. No Overfitting. Precision for positive case is 61% and recall is 2%. The model is terrible at predicting positive cases despite of a much higher accuracy. 

![image](https://github.com/user-attachments/assets/3c6d0f44-1829-40b2-bcb3-94df091ccb87)


## Part 3: Conclusion & Recommendations
1.	Decision Tree model is overfitted. Not a good choice for prediction.
2.	Logistic Regression have 67% accuracy. However, the precision and recall are low, but not terrible. Recommeded for a balanced approach.
3.	ANN have a great accuracy, but terrible at predicting positive cases. If only focus on accuracy, ANN would be recommended. If positive cases are more valued, then logistic regression would be recommended.
## Recommendation
1.	To avoid future default, the bank should focus on people with high earning powers. They tend to have higher educations, more working experiences and better savings. Those features would guarantee them with a low interest rate.
2.	Also Focus on partners. They can share the financial burdens.

## The End
## Thanks For Reading
