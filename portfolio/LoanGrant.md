Bank-loan-prediction
Problem statement

Goal Understand the loan business of the bank. Description A Bank want to understand its loan business. It wants to identify who would be the right target for the loans. The data contains both customers who have been given loans and those who haven’t yet received loans. The aim is to identify of the customers who haven’t been granted a loan which are more likely to repay the loan and provide loans to them. Data 1. Borrower_table - info about borrowers 2. Loan_table – loan details Questions 1. Perform an EDA of the features in the data. Describe any interesting insight you might get when you go through the data. 2. Implement a model to make maximum profit for the bank when providing loans. Please implement atleast 2 algorithms of your choice. Explain the reason for choosing the algorithms.


```python
import numpy as np
import pandas as pd


from sklearn.metrics import auc, roc_curve,roc_auc_score, classification_report,accuracy_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
%matplotlib inline
```


```python
import lightgbm as lgb
```


```python
loan = pd.read_csv('loan_table.csv',parse_dates =[ 'date'])
loan.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19454</td>
      <td>investment</td>
      <td>2012-03-15</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>496811</td>
      <td>investment</td>
      <td>2012-01-17</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>929493</td>
      <td>other</td>
      <td>2012-02-09</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>580653</td>
      <td>other</td>
      <td>2012-06-27</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>172419</td>
      <td>business</td>
      <td>2012-05-21</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
borrower = pd.read_csv('borrower_table.csv')
borrower.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>289774</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8000</td>
      <td>0.49</td>
      <td>3285</td>
      <td>1073</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>482590</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4500</td>
      <td>1.03</td>
      <td>636</td>
      <td>5299</td>
      <td>1</td>
      <td>13500</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135565</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>207797</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1200</td>
      <td>0.82</td>
      <td>358</td>
      <td>3388</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>828078</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6900</td>
      <td>0.80</td>
      <td>2138</td>
      <td>4282</td>
      <td>1</td>
      <td>18100</td>
      <td>36</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
borrower.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 101100 entries, 0 to 101099
    Data columns (total 12 columns):
     #   Column                                           Non-Null Count   Dtype  
    ---  ------                                           --------------   -----  
     0   loan_id                                          101100 non-null  int64  
     1   is_first_loan                                    101100 non-null  int64  
     2   fully_repaid_previous_loans                      46153 non-null   float64
     3   currently_repaying_other_loans                   46153 non-null   float64
     4   total_credit_card_limit                          101100 non-null  int64  
     5   avg_percentage_credit_card_limit_used_last_year  94128 non-null   float64
     6   saving_amount                                    101100 non-null  int64  
     7   checking_amount                                  101100 non-null  int64  
     8   is_employed                                      101100 non-null  int64  
     9   yearly_salary                                    101100 non-null  int64  
     10  age                                              101100 non-null  int64  
     11  dependent_number                                 101100 non-null  int64  
    dtypes: float64(3), int64(9)
    memory usage: 9.3 MB



```python
borrower.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>101100.000000</td>
      <td>101100.000000</td>
      <td>46153.000000</td>
      <td>46153.000000</td>
      <td>101100.000000</td>
      <td>94128.000000</td>
      <td>101100.000000</td>
      <td>101100.000000</td>
      <td>101100.000000</td>
      <td>101100.000000</td>
      <td>101100.000000</td>
      <td>101100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>499666.826726</td>
      <td>0.543492</td>
      <td>0.899291</td>
      <td>0.364332</td>
      <td>4112.743818</td>
      <td>0.724140</td>
      <td>1799.617616</td>
      <td>3177.150821</td>
      <td>0.658675</td>
      <td>21020.727992</td>
      <td>41.491632</td>
      <td>3.864748</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288662.006929</td>
      <td>0.498107</td>
      <td>0.300946</td>
      <td>0.481247</td>
      <td>2129.121462</td>
      <td>0.186483</td>
      <td>1400.545141</td>
      <td>2044.448155</td>
      <td>0.474157</td>
      <td>18937.581415</td>
      <td>12.825570</td>
      <td>2.635491</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>250333.750000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2700.000000</td>
      <td>0.600000</td>
      <td>834.000000</td>
      <td>1706.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>32.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>499885.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4100.000000</td>
      <td>0.730000</td>
      <td>1339.000000</td>
      <td>2673.000000</td>
      <td>1.000000</td>
      <td>21500.000000</td>
      <td>41.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>749706.250000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5500.000000</td>
      <td>0.860000</td>
      <td>2409.000000</td>
      <td>4241.000000</td>
      <td>1.000000</td>
      <td>35300.000000</td>
      <td>50.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999987.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>13500.000000</td>
      <td>1.090000</td>
      <td>10641.000000</td>
      <td>13906.000000</td>
      <td>1.000000</td>
      <td>97200.000000</td>
      <td>79.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
borrower.isnull().sum()
```




    loan_id                                                0
    is_first_loan                                          0
    fully_repaid_previous_loans                        54947
    currently_repaying_other_loans                     54947
    total_credit_card_limit                                0
    avg_percentage_credit_card_limit_used_last_year     6972
    saving_amount                                          0
    checking_amount                                        0
    is_employed                                            0
    yearly_salary                                          0
    age                                                    0
    dependent_number                                       0
    dtype: int64




```python
len(borrower),len(loan)
```




    (101100, 101100)




```python
len(borrower.loan_id.unique())
```




    101100




```python
full_data = pd.merge(left = borrower, right = loan, on = 'loan_id',how = 'left')
full_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>289774</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8000</td>
      <td>0.49</td>
      <td>3285</td>
      <td>1073</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>3</td>
      <td>business</td>
      <td>2012-01-31</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>482590</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4500</td>
      <td>1.03</td>
      <td>636</td>
      <td>5299</td>
      <td>1</td>
      <td>13500</td>
      <td>33</td>
      <td>1</td>
      <td>investment</td>
      <td>2012-11-02</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135565</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>other</td>
      <td>2012-07-16</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>207797</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1200</td>
      <td>0.82</td>
      <td>358</td>
      <td>3388</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>1</td>
      <td>investment</td>
      <td>2012-06-05</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>828078</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6900</td>
      <td>0.80</td>
      <td>2138</td>
      <td>4282</td>
      <td>1</td>
      <td>18100</td>
      <td>36</td>
      <td>1</td>
      <td>emergency_funds</td>
      <td>2012-11-28</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# EDA on Loan Granted


```python
data = full_data[full_data['loan_granted'] == 1]
len(data)
```




    47654




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 47654 entries, 2 to 101099
    Data columns (total 16 columns):
     #   Column                                           Non-Null Count  Dtype         
    ---  ------                                           --------------  -----         
     0   loan_id                                          47654 non-null  int64         
     1   is_first_loan                                    47654 non-null  int64         
     2   fully_repaid_previous_loans                      21865 non-null  float64       
     3   currently_repaying_other_loans                   21865 non-null  float64       
     4   total_credit_card_limit                          47654 non-null  int64         
     5   avg_percentage_credit_card_limit_used_last_year  46751 non-null  float64       
     6   saving_amount                                    47654 non-null  int64         
     7   checking_amount                                  47654 non-null  int64         
     8   is_employed                                      47654 non-null  int64         
     9   yearly_salary                                    47654 non-null  int64         
     10  age                                              47654 non-null  int64         
     11  dependent_number                                 47654 non-null  int64         
     12  loan_purpose                                     47654 non-null  object        
     13  date                                             47654 non-null  datetime64[ns]
     14  loan_granted                                     47654 non-null  int64         
     15  loan_repaid                                      47654 non-null  float64       
    dtypes: datetime64[ns](1), float64(4), int64(10), object(1)
    memory usage: 6.2+ MB



```python
data.isnull().sum()
```




    loan_id                                                0
    is_first_loan                                          0
    fully_repaid_previous_loans                        25789
    currently_repaying_other_loans                     25789
    total_credit_card_limit                                0
    avg_percentage_credit_card_limit_used_last_year      903
    saving_amount                                          0
    checking_amount                                        0
    is_employed                                            0
    yearly_salary                                          0
    age                                                    0
    dependent_number                                       0
    loan_purpose                                           0
    date                                                   0
    loan_granted                                           0
    loan_repaid                                            0
    dtype: int64




```python
# parse date information and extract month, week, and dayofweek information
data['month'] = data['date'].apply(lambda x: x.month)
data['week'] = data['date'].apply(lambda x: x.week)
data['dayofweek'] = data['date'].apply(lambda x: x.dayofweek)
```

    <ipython-input-14-a568e6d2ef8a>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data['month'] = data['date'].apply(lambda x: x.month)
    <ipython-input-14-a568e6d2ef8a>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data['week'] = data['date'].apply(lambda x: x.week)
    <ipython-input-14-a568e6d2ef8a>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data['dayofweek'] = data['date'].apply(lambda x: x.dayofweek)



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>135565</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>other</td>
      <td>2012-07-16</td>
      <td>1</td>
      <td>1.0</td>
      <td>7</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>423171</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>1</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>other</td>
      <td>2012-11-07</td>
      <td>1</td>
      <td>1.0</td>
      <td>11</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200139</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>1</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>business</td>
      <td>2012-09-19</td>
      <td>1</td>
      <td>0.0</td>
      <td>9</td>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>991294</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>1</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>emergency_funds</td>
      <td>2012-12-04</td>
      <td>1</td>
      <td>1.0</td>
      <td>12</td>
      <td>49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>875332</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>1</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>business</td>
      <td>2012-01-20</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Data quality check
data[(data.is_first_loan == 1 ) & (data.fully_repaid_previous_loans == 1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
data[(data.is_first_loan == 1 ) & (data.currently_repaying_other_loans == 1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#Visualization of loan purpose

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.countplot(x='loan_purpose', data=data, ax=ax[0])
ax[0].set_xlabel('Loan Purpose', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of Loan Purpose', fontsize=16)

sns.barplot(x='loan_purpose', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('Loan Purpose', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. Loan Purpose', fontsize=16)
plt.tight_layout()
plt.show()

```


![png](output_19_0.png)



```python
#Dependent number
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.countplot(x='dependent_number', data=data, ax=ax[0])
ax[0].set_xlabel('dependent_number', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of dependent_number', fontsize=16)

sns.barplot(x='dependent_number', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('dependent_number', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. dependent_number', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_20_0.png)



```python
#Age
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.countplot(x='age', data=data, ax=ax[0])
ax[0].set_xlabel('age', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of age', fontsize=16)

sns.barplot(x='age', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('age', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. age', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_21_0.png)



```python
#is_employed
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.countplot(x='is_employed', data=data, ax=ax[0])
ax[0].set_xlabel('is_employed', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of is_employed', fontsize=16)

sns.barplot(x='is_employed', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('is_employed', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. is_employed', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_22_0.png)



```python
#is_first_loan

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.countplot(x='is_first_loan', data=data, ax=ax[0])
ax[0].set_xlabel('is_first_loan', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of is_first_loan', fontsize=16)

sns.barplot(x='is_first_loan', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('is_first_loan', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. is_first_loan', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_23_0.png)



```python
#fully_repaid_previous_loans
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.countplot(x='fully_repaid_previous_loans', data=data, ax=ax[0])
ax[0].set_xlabel('fully_repaid_previous_loans', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of fully_repaid_previous_loans', fontsize=16)

sns.barplot(x='fully_repaid_previous_loans', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('fully_repaid_previous_loans', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. fully_repaid_previous_loans', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_24_0.png)



```python
#currently_repaying_other_loans
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.countplot(x='currently_repaying_other_loans', data=data, ax=ax[0])
ax[0].set_xlabel('currently_repaying_other_loans', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of currently_repaying_other_loans', fontsize=16)

sns.barplot(x='currently_repaying_other_loans', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('currently_repaying_other_loans', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. currently_repaying_other_loans', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_25_0.png)



```python
# Visualization of 'total_credit_card_limit'
grouped = data[['total_credit_card_limit', 'loan_repaid']].groupby('total_credit_card_limit')
grouped
mean = grouped.mean().reset_index()
mean
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.distplot(data[data['loan_repaid'] == 0]['total_credit_card_limit'], 
             label='loan_repaid = 0', ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['loan_repaid'] == 1]['total_credit_card_limit'], 
             label='loan_repaid = 1', ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Histogram of total_credit_card_limit', fontsize=16)
ax[0].legend()
ax[1].plot(mean['total_credit_card_limit'], mean['loan_repaid'], '.-')
ax[1].set_title('Mean Repaid Rate vs. total_credit_card_limit', fontsize=16)
ax[1].set_xlabel('total_credit_card_limit')
ax[1].set_ylabel('Mean Repaid Rate')
ax[1].grid(True)
plt.show()
```


![png](output_26_0.png)



```python
# Visualization of 'saving_amount'
grouped = data[['saving_amount', 'loan_repaid']].groupby('saving_amount')
grouped
mean = grouped.mean().reset_index()
mean
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.distplot(data[data['loan_repaid'] == 0]['saving_amount'], 
             label='loan_repaid = 0', ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['loan_repaid'] == 1]['saving_amount'], 
             label='loan_repaid = 1', ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Histogram of saving_amount', fontsize=16)
ax[0].legend()
ax[1].plot(mean['saving_amount'], mean['loan_repaid'], '.-')
ax[1].set_title('Mean Repaid Rate vs. saving_amount', fontsize=16)
ax[1].set_xlabel('saving_amount')
ax[1].set_ylabel('Mean Repaid Rate')
ax[1].grid(True)
plt.show()
```


![png](output_27_0.png)



```python
# Visualization of 'checking_amount'
grouped = data[['checking_amount', 'loan_repaid']].groupby('checking_amount')
grouped
mean = grouped.mean().reset_index()
mean
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.distplot(data[data['loan_repaid'] == 0]['checking_amount'], 
             label='loan_repaid = 0', ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['loan_repaid'] == 1]['checking_amount'], 
             label='loan_repaid = 1', ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Histogram of checking_amount', fontsize=16)
ax[0].legend()
ax[1].plot(mean['checking_amount'], mean['loan_repaid'], '.-')
ax[1].set_title('Mean Repaid Rate vs. checking_amount', fontsize=16)
ax[1].set_xlabel('checking_amount')
ax[1].set_ylabel('Mean Repaid Rate')
ax[1].grid(True)
plt.show()
```


![png](output_28_0.png)



```python
# Visualization of 'yearly_salry'
grouped = data[['yearly_salary', 'loan_repaid']].groupby('yearly_salary')
grouped
mean = grouped.mean().reset_index()
mean
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.distplot(data[data['loan_repaid'] == 0]['yearly_salary'], 
             label='loan_repaid = 0', ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['loan_repaid'] == 1]['yearly_salary'], 
             label='loan_repaid = 1', ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Histogram of yearly_salary', fontsize=16)
ax[0].legend()
ax[1].plot(mean['yearly_salary'], mean['loan_repaid'], '.-')
ax[1].set_title('Mean Repaid Rate vs. yearly_salary', fontsize=16)
ax[1].set_xlabel('yearly_salary')
ax[1].set_ylabel('Mean Repaid Rate')
ax[1].grid(True)
plt.show()
```


![png](output_29_0.png)



```python
not_null = data[~data['avg_percentage_credit_card_limit_used_last_year'].isnull()]
```


```python
#avg_percentage_credit_card_limit_used_last_year


grouped = not_null[['avg_percentage_credit_card_limit_used_last_year', 'loan_repaid']].groupby('avg_percentage_credit_card_limit_used_last_year')
grouped
mean = grouped.mean().reset_index()
mean

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.04</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.05</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>104</th>
      <td>1.05</td>
      <td>0.279070</td>
    </tr>
    <tr>
      <th>105</th>
      <td>1.06</td>
      <td>0.342466</td>
    </tr>
    <tr>
      <th>106</th>
      <td>1.07</td>
      <td>0.310811</td>
    </tr>
    <tr>
      <th>107</th>
      <td>1.08</td>
      <td>0.258065</td>
    </tr>
    <tr>
      <th>108</th>
      <td>1.09</td>
      <td>0.416667</td>
    </tr>
  </tbody>
</table>
<p>109 rows × 2 columns</p>
</div>




```python
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.distplot(not_null[not_null['loan_repaid'] == 0]['avg_percentage_credit_card_limit_used_last_year'], 
             label='loan_repaid = 0', ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['loan_repaid'] == 1]['avg_percentage_credit_card_limit_used_last_year'], 
             label='loan_repaid = 1', ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Histogram of avg_percentage_credit_card_limit_used_last_year', fontsize=16)
ax[0].legend()
ax[1].plot(mean['avg_percentage_credit_card_limit_used_last_year'], mean['loan_repaid'], '.-')
ax[1].set_title('Mean Repaid Rate vs. avg_percentage_credit_card_limit_used_last_year', fontsize=16)
ax[1].set_xlabel('avg_percentage_credit_card_limit_used_last_year')
ax[1].set_ylabel('Mean Repaid Rate')
ax[1].grid(True)
plt.show()
```


![png](output_32_0.png)



```python
#month
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
sns.countplot(x='month', data=data, ax=ax[0])
ax[0].set_xlabel('month', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of month', fontsize=16)

sns.barplot(x='month', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('month', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. month', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_33_0.png)



```python
#dayofweek
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
sns.countplot(x='dayofweek', data=data, ax=ax[0])
ax[0].set_xlabel('dayofweek', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].set_title('Count Plot of dayofweek', fontsize=16)

sns.barplot(x='dayofweek', y='loan_repaid', data=data, ax=ax[1])
ax[1].set_xlabel('dayofweek', fontsize=12)
ax[1].set_ylabel('Loan Repaid Ratio', fontsize=12)
ax[1].set_title('Loan Repaid Ratio vs. dayofweek', fontsize=16)
plt.tight_layout()
plt.show()
```


![png](output_34_0.png)


# Feature Engineering to predict if loan will be repaid or not.


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>135565</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>other</td>
      <td>2012-07-16</td>
      <td>1</td>
      <td>1.0</td>
      <td>7</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>423171</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>1</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>other</td>
      <td>2012-11-07</td>
      <td>1</td>
      <td>1.0</td>
      <td>11</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200139</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>1</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>business</td>
      <td>2012-09-19</td>
      <td>1</td>
      <td>0.0</td>
      <td>9</td>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>991294</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>1</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>emergency_funds</td>
      <td>2012-12-04</td>
      <td>1</td>
      <td>1.0</td>
      <td>12</td>
      <td>49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>875332</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>1</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>business</td>
      <td>2012-01-20</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(data[data['is_first_loan'] == 1]['fully_repaid_previous_loans'].unique())
print(data[data['is_first_loan'] == 1]['currently_repaying_other_loans'].unique())
```

    [nan]
    [nan]



```python
data = data.fillna({'fully_repaid_previous_loans': -1,'currently_repaying_other_loans':-1 })
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>135565</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>other</td>
      <td>2012-07-16</td>
      <td>1</td>
      <td>1.0</td>
      <td>7</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>423171</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>1</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>other</td>
      <td>2012-11-07</td>
      <td>1</td>
      <td>1.0</td>
      <td>11</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200139</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>1</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>business</td>
      <td>2012-09-19</td>
      <td>1</td>
      <td>0.0</td>
      <td>9</td>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>991294</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>1</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>emergency_funds</td>
      <td>2012-12-04</td>
      <td>1</td>
      <td>1.0</td>
      <td>12</td>
      <td>49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>875332</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>1</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>business</td>
      <td>2012-01-20</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
median_cc = data['avg_percentage_credit_card_limit_used_last_year'].median()
data = data.fillna({'avg_percentage_credit_card_limit_used_last_year':median_cc})
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>date</th>
      <th>loan_granted</th>
      <th>loan_repaid</th>
      <th>month</th>
      <th>week</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>135565</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>other</td>
      <td>2012-07-16</td>
      <td>1</td>
      <td>1.0</td>
      <td>7</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>423171</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>1</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>other</td>
      <td>2012-11-07</td>
      <td>1</td>
      <td>1.0</td>
      <td>11</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200139</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>1</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>business</td>
      <td>2012-09-19</td>
      <td>1</td>
      <td>0.0</td>
      <td>9</td>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>991294</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>1</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>emergency_funds</td>
      <td>2012-12-04</td>
      <td>1</td>
      <td>1.0</td>
      <td>12</td>
      <td>49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>875332</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>1</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>business</td>
      <td>2012-01-20</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove date fields
data = data.drop(columns = ['month','week','dayofweek','date'],axis = 1)
```


```python
data_cp = data.drop(columns = ['loan_granted','loan_id'],axis = 1)
```


```python
data_cp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>is_employed</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_purpose</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>1</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>other</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>1</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>other</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>1</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>business</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>1</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>emergency_funds</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>1</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>business</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(data[data['is_employed'] == 0]['yearly_salary'].unique())
#Co-related metrics. Remove one
```

    [0]



```python
data_cp = data_cp.drop(columns = ['is_employed'],axis = 1)
```


```python
data_cp.loan_purpose.value_counts()
```




    home               11614
    business           10773
    investment         10603
    emergency_funds     7753
    other               6911
    Name: loan_purpose, dtype: int64




```python
X = pd.get_dummies(data_cp)
X.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_repaid</th>
      <th>loan_purpose_business</th>
      <th>loan_purpose_emergency_funds</th>
      <th>loan_purpose_home</th>
      <th>loan_purpose_investment</th>
      <th>loan_purpose_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = X.drop(columns = ['loan_purpose_other'],axis = 1)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>is_first_loan</th>
      <th>fully_repaid_previous_loans</th>
      <th>currently_repaying_other_loans</th>
      <th>total_credit_card_limit</th>
      <th>avg_percentage_credit_card_limit_used_last_year</th>
      <th>saving_amount</th>
      <th>checking_amount</th>
      <th>yearly_salary</th>
      <th>age</th>
      <th>dependent_number</th>
      <th>loan_repaid</th>
      <th>loan_purpose_business</th>
      <th>loan_purpose_emergency_funds</th>
      <th>loan_purpose_home</th>
      <th>loan_purpose_investment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6900</td>
      <td>0.82</td>
      <td>2085</td>
      <td>3422</td>
      <td>24500</td>
      <td>38</td>
      <td>8</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6100</td>
      <td>0.53</td>
      <td>6163</td>
      <td>5298</td>
      <td>29500</td>
      <td>24</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4000</td>
      <td>0.57</td>
      <td>602</td>
      <td>2757</td>
      <td>31700</td>
      <td>36</td>
      <td>8</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7000</td>
      <td>0.52</td>
      <td>2575</td>
      <td>2917</td>
      <td>58900</td>
      <td>33</td>
      <td>3</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4300</td>
      <td>0.83</td>
      <td>722</td>
      <td>892</td>
      <td>5400</td>
      <td>32</td>
      <td>7</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# LightGBM Model


```python
#Train test split
Y = X['loan_repaid']
Y.head()
```




    2    1.0
    5    1.0
    7    0.0
    8    1.0
    9    1.0
    Name: loan_repaid, dtype: float64




```python
X = X.drop(columns= ['loan_repaid'])

```


```python
len(X)
```




    47654




```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
```


```python
len(X_train),len(X_test),len(y_train),len(y_test)
```




    (35740, 11914, 35740, 11914)




```python
#create LightGBM dataset
d_train = lgb.Dataset(data=X_train, label=y_train,free_raw_data=False)
```


```python
# Cross validation
params = {'learning_rate': 0.01, 
          'boosting_type': 'gbdt', 
          'objective': 'binary', 
          'metric': ['binary_logloss', 'auc'], 
          'sub_feature':0.5, 
          'num_leaves': 31, 
          'min_data': 50, 
          'max_depth': 30, 
          'is_unbalance': True}

history = lgb.cv(params, train_set=d_train, num_boost_round=1000, nfold=5, 
                 early_stopping_rounds=20, seed=42, verbose_eval=False)

print('Best rounds:\t', len(history['auc-mean']))
```

    Best rounds:	 822



```python
# re-train the model and make predictions
clf = lgb.train(params, train_set=d_train, num_boost_round=822)
pred = clf.predict(X_test)
```


```python
# feature importance
features = clf.feature_name()
importance = clf.feature_importance()

fig, ax = plt.subplots(figsize=(10, 8))
lgb.plot_importance(clf, ax=ax, height=0.5)
plt.show()
```


![png](output_57_0.png)



```python
#Total Profit from original Bank model

loan_paid = y_test.astype(int).values
bank_profit = np.sum(loan_paid * 2 - 1)
print('Bank profit:\t', bank_profit)
```

    Bank profit:	 3390



```python
# Calculate profit according to different thresholds
def calc_threshold(loan_paid,probability,threshold):
    loan_granted = (probability > threshold).astype(int)
    profit = 0
    for i in range(len(loan_paid)):
        if loan_granted[i] == 1:
            if loan_paid[i] == 0:
                profit -= 1
            else:
                profit += 1
                
    return profit
        
            
    
```


```python
# calculate the profit according to given threshold
thresholds = list(np.linspace(0, 1, 100))
profits = []

for threshold in thresholds:
    profits.append(calc_threshold(loan_paid, pred, threshold))
```


```python
thresh_prof = pd.DataFrame({'threshold':thresholds,'profit':profits})
thresh_prof.loc[thresh_prof.profit.argmax(),:]
```




    threshold       0.40404
    profit       6762.00000
    Name: 40, dtype: float64




```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(thresholds, profits, label='New Model')
ax.plot(thresholds, [bank_profit] * len(thresholds), label='Bank Model')
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Profit', fontsize=12)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()
```


![png](output_62_0.png)



```python

fpr,tpr,threshold = roc_curve(loan_paid, pred, pos_label=1)
print("Area under the Receiver Operating Characteristic Curves (ROC AUC) is %f " 
      %(roc_auc_score(loan_paid, pred)))


```

    Area under the Receiver Operating Characteristic Curves (ROC AUC) is 0.977675 



```python
plt.figure()
plt.plot([0,1],[0,1], linestyle = '--', label = 'random guessing')
plt.plot(fpr, tpr, label = "ROC AUC" )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
```




    <matplotlib.legend.Legend at 0x127da9eb0>




![png](output_64_1.png)



```python
y_pred = (pred > 0.4).astype(int)
y_pred
```




    array([0, 0, 0, ..., 0, 1, 0])




```python
len(y_pred)

```




    11914




```python

print(classification_report(y_true = loan_paid, y_pred=y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.88      0.92      0.90      4262
               1       0.96      0.93      0.94      7652
    
        accuracy                           0.93     11914
       macro avg       0.92      0.92      0.92     11914
    weighted avg       0.93      0.93      0.93     11914
    


Answer Question 3 , "Describe the impact of the most important variables on the prediction. Also, focus on the variable "is_employed", which describes whether the borrower is employed when she asks for the loan. How does this variable impact the model? Explain why"

according to above plot, the most important feature is 'saving amount'. This makes perfect sense. Money in the saving account, meaure a person's ability to control his financial situation. More money in the saving account, more likely the borrower can repay.

as I mentioned, in my model, 'is_employed' is merged into feature 'salary', which correspond to the case 'salary=0'. according to above plot, salary is also a strong feature.

if not employed, then salary=0, which will impact negatively on 'grant decision'. that is very easy to understand, a person without any financial input, will be very unlikely to repay.
but even the person is employed, whether grant the loan or not, depends on his salary. higher salary indicates strong ability to repay.
