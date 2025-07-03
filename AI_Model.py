#!/usr/bin/env python
# coding: utf-8

# # Reading Files

# In[435]:


import pandas as pd
import numpy as np

oct_data = pd.read_excel('Transactions (Oct 2024).xlsx')
nov_data = pd.read_excel('Transactions (Nov 2024).xlsx')

transaction = pd.concat([oct_data, nov_data], ignore_index=True)


# In[230]:


costumers = pd.read_excel('Customer Data.xlsx')
corporate = costumers[costumers['CUSTOMER TYPE'] == 'CORPORATE']


# In[436]:


costumers['Suspecious Transactions']=0


# In[236]:


compare = pd.read_excel('Copy of Re-downloaded Online - Transaction Data_v2.xlsx')


# In[437]:


october = compare[:881] #this is the blocked transactions from only october and novermber
october['Block Date'].max()


# # Filtering Suspecious Clients

# ## FIRST PROBLEM

# ### 6 transactions not blocked

# In[438]:


transaction['FULL NAME'] = transaction['SEN_FIRST_NAME'].astype(str) + ' ' + transaction['SEN_LAST_NAME'].astype(str)


# In[ ]:


#Filtering any corporate remittance transaction Amounted to AED 500,000 and above
amount_500 = transaction[transaction['TOTAL_AMT'] >= 500000]
flagged = amount_500[amount_500['CARD_ID'].isin(corporate['CARD_NO'])] # count 102

#Filtering the blocked transaction for this specific rule which are 96
blocked = october[october['Block Reason'] == 'Corporate Customer above Sending Limit 500,000'] # count 96


print("This 6 transactions were flagged by me and they were not in the blocked transactions")
flagged[~flagged['FTRN'].isin(blocked['Ref Ftrn'])][['FTRN','LCY_AMT','TOTAL_AMT']] # count 6


# In[440]:


blocked[blocked['Ref Ftrn'] == 5610110908958]
#Proving that they aren't in the blocked list


# In[441]:


costumers.loc[costumers['CARD_NO'].isin(flagged['CARD_ID']), 'Suspecious Transactions'] = 1


# ### proving the correct 96 values

# In[569]:


#flagged[:3] # Proving the other 96 flagged values are in the 96 blocked


# In[570]:


#blocked[blocked['Ref Ftrn'] == 5623100003752]


# In[571]:


#blocked[blocked['Ref Ftrn'] == 5620340001915]


# In[572]:


#blocked[blocked['Ref Ftrn'] == 5620310188003]


# ## SECOND PROBLEM

# ### 45 transactions (if LCY >300,000 and individual) not blocked

# In[573]:


#filtering for high/medium/low risk and above sending limit of 300,000
blocked_2 = october[october['Block Reason'] == 'Individual Customer with Low, Medium and Medium High Risk Rating above Sending Limit\n300,000']
individuals = costumers[costumers['CUSTOMER TYPE'] == 'INDIVIDUAL']

#The amount of blocked transactions for this reason was 173
amount_300 = transaction[transaction['LCY_AMT'] >= 300000]
flagged_2 = amount_300[amount_300['CARD_ID'].isin(individuals['CARD_NO'])] #count 218

#All of this 57 transactions where flagged by me and not in the blocked transactions (TOTAL_AMOUNT > 300,000 && INDIVIDUAL). 57
#flagged_2[~flagged_2['FTRN'].isin(blocked_2['Ref Ftrn'])][['FTRN','LCY_AMT','TOTAL_AMT']]


# In[574]:


#blocked_2[blocked_2['Ref Ftrn'] == 5610110908634]
#Proving that they aren't in the blocked list


# In[448]:


costumers.loc[costumers['CARD_NO'].isin(flagged_2['CARD_ID']), 'Suspecious Transactions'] = 1


# ### proving the other values are in the blocked list

# In[579]:


#using a sample of the first three transactions which LCY > 300000 and INDIVIDUAL
flagged_2[flagged_2['FTRN'].isin(blocked_2['Ref Ftrn'])][['FTRN','LCY_AMT']][:3]
print()


# In[580]:


blocked_2[blocked_2['Ref Ftrn'] == 5620120001799]
#Proving that they in the blocked list
print()


# In[581]:


blocked_2[blocked_2['Ref Ftrn'] == 5610340543845]
print()


# In[582]:


blocked_2[blocked_2['Ref Ftrn'] == 5620640019351]
print()


# ## THIRD PROBLEM

# ### I flagged 14. There are 23 blocked transactions. I wasn't able to flag 11 of them but found 2 new ones

# In[575]:


#Individual Customer with High Risk Rating above Sending Limit 200,000

#There are 23 blocked transactions
blocked_3 = october[october['Block Reason']== 'Individual Customer with High Risk Rating above Sending Limit 200,000']

#filtering for individuals of high risk
indivi = costumers[costumers['CUSTOMER TYPE'] == 'INDIVIDUAL']
individual = indivi[indivi['Risk Rating'] == 'HIGH']

#filtering for lcy_amt >200000 found only 14. There are 11 blocked transactions that I wasn't able to find.
#meaning there are 2 transactions I flagged that weren't blocked
amount_200 = transaction[transaction['LCY_AMT'] >= 200000]
flagged_3 = amount_200[amount_200['CARD_ID'].isin(individual['CARD_NO'])] 

#Checking the 11 blocked transactions that I wasn't able to find
not_flagged = blocked_3[~blocked_3['Ref Ftrn'].isin(flagged_3['FTRN'])]
#not_flagged


# In[454]:


costumers.loc[costumers['CARD_NO'].isin(flagged_3['CARD_ID']), 'Suspecious Transactions'] = 1


# ### proving why  wasn't able to find 11 blocked transactions

# In[576]:


individual_blocked = indivi[indivi['CARD_NO'].isin(not_flagged['Card Id'])]
individual_blocked[['CARD_NO','Risk Rating','CUSTOMER TYPE']]
#You will see that their risk rating is not high.
#meaning that when the transactions got blocked they were high risk but now they are not. That's why I couldn't flag them
print()


# ## FOURTH PROBLEM

# In[577]:


#Corporate costumer transaction in PKR currency
blocked_4 = october[october['Block Reason'] == 'Corporate to PKR']
corp = costumers[costumers['CUSTOMER TYPE'] == 'CORPORATE']

#There were 85 blocked and I flagged 85
pkr_check = transaction[transaction['CARD_ID'].isin(corp['CARD_NO'])] 
flagged_4 = pkr_check[pkr_check['CCY'] == 'PAKISTAN RUPEES'] # 85

#flagged_4[~flagged_4['FTRN'].isin(blocked_4['Ref Ftrn'])][['FTRN','LCY_AMT','TOTAL_AMT']]
flagged_4
print()


# In[578]:


flagged_4[['CCY','FTRN']]
print()


# In[458]:


costumers.loc[costumers['CARD_NO'].isin(flagged_4['CARD_ID']), 'Suspecious Transactions'] = 1


# # AI Model

# In[532]:


suspecious_1 = costumers[costumers['Suspecious Transactions'] == 1]
suspecious_0 = costumers[costumers['Suspecious Transactions'] == 0]

suspecious_0_sampled = suspecious_0.sample(n=len(suspecious_1), random_state=42)
balanced_data = pd.concat([suspecious_1, suspecious_0_sampled])

balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)


# In[536]:


# This AI model will use costumer's information to predict if they will perform a suspecious transaction
y = balanced_data[['Suspecious Transactions']] #442 values thar have suspecious transactions
X = balanced_data[['CARD_NO','Nationality','Risk Rating','BI_RATING','EXPECTED_ANNUAL_VALUE','EXPECTED_ANNUAL_TRX','RESIDENT STATUS','OCCUPATION','CUSTOMER TYPE','country_of_birth','CARD STATUS']]

#Columns dropped: CORP_BANK_CODE, Economic Activity, Company Type, FPEP

# CORP_BANK_CODE were different values for each costumer so it has no value to AI training
# Economic Activity, Company Type Has 396623 values which are 'NaN' and 1979 which are actual values. It does more harm than good
# FPEP has NO values


# ### Data Filtering

# In[537]:


#Each client should pick which countries are high-risk for them
high_risk_countries = [
    'PAKISTAN', 'BANGLADESH', 'EGYPT', 'PALESTINE', 'PHILIPPINES', 'NIGERIA', 'YEMEN', 
    'SUDAN', 'IRAQ', 'TUNISIA', 'MOROCCO', 'NIGER', 'ETHIOPIA', 'SYRIA', 'AFGHANISTAN', 
    'VENEZUELA', 'COLOMBIA', 'MYANMAR', 'IRAN', 'BURUNDI', 'DOMINICAN REPUBLIC', 
    'SOUTH SUDAN', 'RWANDA', 'CUBA', 'SOMALIA', 'SOUTH AFRICA', 'LIBERIA', 'SIERRA LEONE', 
    'MALI', 'CHAD', 'SOMALIA', 'MAURITANIA', 'CONGO DEM. REP.(brazzaville)', 
    'CENTRAL AFRICAN REPUBLIC', 'LIBYA', 'GUINEA', 'GUINEA-BISSAU'
]

X.loc[:,'Nationality'] = X['Nationality'].map(lambda x: 1 if x in high_risk_countries else 0)


# In[538]:


X.loc[:,'Risk Rating'] = X['Risk Rating'].map({'LOW':1, 'MEDIUM':2, 'HIGH':3})
X.loc[:,'BI_RATING'] = X['BI_RATING'].map({'LOW':1, 'MEDIUM':2, 'HIGH':3})


# In[539]:


def midpoint(annual_ven):
    if 'and above' in annual_ven:
        return int(annual_ven.split()[0]) + 5000
    else:
        left, right = map(int, annual_ven.split('-'))
        mid = (left + right)/2
        return mid

X.loc[:,'EXPECTED_ANNUAL_VALUE'] = X['EXPECTED_ANNUAL_VALUE'].apply(midpoint)


# In[540]:


def midpoint(annual_ven):
    if 'and above' in annual_ven:
        return int(annual_ven.split()[0]) + 10
    else:
        left, right = map(int, annual_ven.split('-'))
        mid = (left + right)/2
        return mid
X.loc[:,'EXPECTED_ANNUAL_TRX'] = X['EXPECTED_ANNUAL_TRX'].apply(midpoint)


# In[541]:


X.loc[:,'RESIDENT STATUS'] = X['RESIDENT STATUS'].map({'UAE-RESIDENT':1, 'NON-RESIDENT':0})


# In[542]:


high_occupation = ['TELLER/CASHIER', 'BANKER', 'ACCOUNTING CLERK', 
                        'FINANCE MANAGER', 'INVESTOR', 'LAWYER']
medium_occupation = ['TEACHER', 'NURSE', 'PAINTER', 'DOCTOR', 'ACTOR', 
                     'ENGINEER', 'CHEMICAL PRODUCTS MACHINE OPERATOR']
X.loc[:,'OCCUPATION'] = X['OCCUPATION'].map(lambda x: 3 if x in high_occupation else(2 if x in medium_occupation else 1))


# In[543]:


X.loc[:,'CUSTOMER TYPE'] = X['CUSTOMER TYPE'].map({'INDIVIDUAL':1, 'CORPORATE':0})


# In[544]:


X.loc[:,'CARD STATUS'] = X['CARD STATUS'].map({'IN-ACTIVE':1, 'ACTIVE':0})


# In[545]:


high_risk_countries = ['NEPAL', 'PAKISTAN', 'INDIA', 'BANGLADESH', 'EGYPT', 'LEBANON',
                       'UNITED ARAB EMIRATES', 'PHILIPPINES', 'NIGERIA', 'CHINA', 'YEMEN', 
                       'UGANDA', 'SUDAN', 'KENYA', 'IRAQ', 'TUNISIA', 'MOROCCO', 'NIGER', 
                       'ETHIOPIA', 'JORDAN', 'ZIMBABWE', 'INDONESIA', 'SRILANKA', 'SAUDI ARABIA',
                       'ALGERIA', 'SYRIA', 'IRAN', 'TANZANIA', 'PALESTINE', 'AFGHANISTAN', 
                       'SOMALIA', 'CONGO DEM. REP.(brazzaville)', 'MALI', 'LIBYA', 'GUINEA', 
                       'SOUTH SUDAN', 'BURKINA FASO', 'CAMBODIA', 'ANGOLA', 'BURUNDI', 'CUBA']

medium_risk_countries = ['UNITED KINGDOM', 'FRANCE', 'GERMANY', 'AUSTRALIA', 'UNITED STATES OF AMERICA',
                         'CANADA', 'SPAIN', 'ITALY', 'SWEDEN', 'GREECE', 'PORTUGAL', 'BELGIUM', 'POLAND',
                         'NETHERLANDS', 'SWITZERLAND', 'DENMARK', 'FINLAND', 'IRELAND', 'NORWAY', 'AUSTRIA']
X.loc[:,'country_of_birth'] = X['country_of_birth'].map(lambda x: 3 if x in high_risk_countries else(2 if x in medium_risk_countries else 1))


# ### Training

# In[546]:


from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(X,y,random_state=1)


# In[547]:


train_id = train_x['CARD_NO']
val_id = val_x['CARD_NO']
train_x = train_x.drop('CARD_NO', axis=1)
val_x = val_x.drop('CARD_NO', axis=1)


# In[548]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
X_train = scaler.fit_transform(train_x)
X_test = scaler.transform(val_x)


# In[549]:


model = LogisticRegression()

model.fit(X_train, train_y)


# # Output

# In[557]:


y_pred = model.predict(X_test)

accuracy = accuracy_score(val_y, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[551]:


val_id_flat = val_id.to_numpy().flatten()
val_y_flat = val_y.to_numpy().flatten()
y_pred_flat = y_pred.flatten()
df = pd.DataFrame({'Customer': val_id_flat, 'Actual': val_y_flat, 'Predicted': y_pred_flat})

