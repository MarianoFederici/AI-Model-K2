# AI-Model-K2
I developed an AI model using Jupyter, Python, pandas, and scikit-learn to predict high-risk clients based on Suspicious Transaction Reports (STRs) for financial institutions. For security purposes and client data protection I changed the file from a jupiter notebook to a python file.

Step-by-Step Process:
  1. Data Collection:I began by loading transaction data from an exchange house based in the UAE for the months of October and November. Additionally, I
     extracted records of blocked transactions by the exchange house for further analysis.
  2. Data Analysis & Verification:I conducted a thorough analysis of the blocked transactions, validating whether the exchange house had appropriately flagged
     transactions as suspicious. The key filtering criteria included:
         Corporate remittance transactions exceeding AED 500,000
         High, medium, and low-risk transactions above the sending limit of AED 300,000
         Individual customers with high-risk ratings exceeding a sending limit of AED 200,000
         Corporate customers with transactions in PKR currency
  3. Labeling the Data: For each transaction, I marked whether the customer had been flagged for suspicious activity (STR). This was the target variable for
     model training.
  4. Data Preprocessing:I cleaned and prepared the dataset by filtering out irrelevant or redundant columns and ensuring the data was structured appropriately
     for the learning algorithm. Applied StandardScaler from scikit-learn to scale the numerical features for better model performance.Split the dataset into
     training and validation sets to ensure unbiased evaluation and later accuracy assessment.
  5. Model Development: I selected Logistic Regression as the model of choice due to its effectiveness in binary classification tasks, particularly for
     predicting whether a transaction is suspicious or not.
  6. Model Training & Evaluation: I trained the logistic regression model on the training data and evaluated its performance on the validation set. Using
     accuracy_score from scikit-learn, I measured the model’s accuracy, which resulted in an impressive 80.09% accuracy.
