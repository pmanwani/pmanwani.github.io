---
layout: post
title:  "Item Clustering by User Purchase History"
date:   2020-08-23 15:03:14 -0400
categories: jekyll update
---

<font color="grey" size = 4><i>Skills: K-means Clustering, PCA, Silhouette Score </i></font>

We have two datasets:<br>
1. ***item_to_id*** which has information on the item and it's corresponding ID to uniquely identify each product.<br>
2. ***purchase_history*** which contains information on user's purchase history. The products users tend to buy together.<br>
We can use these datasets to come up with interesting insights that may help expand the business.

### Business Questions:

#### Target Audience
1. Identify top 10 customers who bought the most items overall
2. For each item, identify the customer who bought that product the most

This will help us identify our most valuable and loyal customers who can then advocate for our business.

#### Cluster Items

1. Cluster items based on user co-purchase history. That is, create clusters of products that have the highest probability of being bought together. The goal of this is to replace the old/manually created categories with these new ones. Each item can belong to just one cluster.




```python
import re
from collections import Counter
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
```

### Load and Inspect Data

There are 48 unique items in the inventory.


```python
item = pd.read_csv("item_to_id.csv")
item.head()
item.sort_values(by = 'Item_id')
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
      <th>Item_name</th>
      <th>Item_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>sugar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>lettuce</td>
      <td>2</td>
    </tr>
    <tr>
      <th>47</th>
      <td>pet items</td>
      <td>3</td>
    </tr>
    <tr>
      <th>46</th>
      <td>baby items</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>waffles</td>
      <td>5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>poultry</td>
      <td>6</td>
    </tr>
    <tr>
      <th>41</th>
      <td>sandwich bags</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>butter</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>soda</td>
      <td>9</td>
    </tr>
    <tr>
      <th>32</th>
      <td>carrots</td>
      <td>10</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cereals</td>
      <td>11</td>
    </tr>
    <tr>
      <th>42</th>
      <td>shampoo</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bagels</td>
      <td>13</td>
    </tr>
    <tr>
      <th>12</th>
      <td>eggs</td>
      <td>14</td>
    </tr>
    <tr>
      <th>40</th>
      <td>aluminum foil</td>
      <td>15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>milk</td>
      <td>16</td>
    </tr>
    <tr>
      <th>24</th>
      <td>beef</td>
      <td>17</td>
    </tr>
    <tr>
      <th>36</th>
      <td>laundry detergent</td>
      <td>18</td>
    </tr>
    <tr>
      <th>45</th>
      <td>shaving cream</td>
      <td>19</td>
    </tr>
    <tr>
      <th>29</th>
      <td>grapefruit</td>
      <td>20</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cheeses</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>frozen vegetables</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tea</td>
      <td>23</td>
    </tr>
    <tr>
      <th>38</th>
      <td>paper towels</td>
      <td>24</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cherries</td>
      <td>25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>spaghetti sauce</td>
      <td>26</td>
    </tr>
    <tr>
      <th>37</th>
      <td>dishwashing</td>
      <td>27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>canned vegetables</td>
      <td>28</td>
    </tr>
    <tr>
      <th>44</th>
      <td>hand soap</td>
      <td>29</td>
    </tr>
    <tr>
      <th>17</th>
      <td>flour</td>
      <td>30</td>
    </tr>
    <tr>
      <th>19</th>
      <td>pasta</td>
      <td>31</td>
    </tr>
    <tr>
      <th>30</th>
      <td>apples</td>
      <td>32</td>
    </tr>
    <tr>
      <th>39</th>
      <td>toilet paper</td>
      <td>33</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tortillas</td>
      <td>34</td>
    </tr>
    <tr>
      <th>43</th>
      <td>soap</td>
      <td>35</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ice cream</td>
      <td>36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dinner rolls</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>juice</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sandwich loaves</td>
      <td>39</td>
    </tr>
    <tr>
      <th>27</th>
      <td>berries</td>
      <td>40</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ketchup</td>
      <td>41</td>
    </tr>
    <tr>
      <th>34</th>
      <td>cucumbers</td>
      <td>42</td>
    </tr>
    <tr>
      <th>0</th>
      <td>coffee</td>
      <td>43</td>
    </tr>
    <tr>
      <th>31</th>
      <td>broccoli</td>
      <td>44</td>
    </tr>
    <tr>
      <th>33</th>
      <td>cauliflower</td>
      <td>45</td>
    </tr>
    <tr>
      <th>26</th>
      <td>bananas</td>
      <td>46</td>
    </tr>
    <tr>
      <th>25</th>
      <td>pork</td>
      <td>47</td>
    </tr>
    <tr>
      <th>14</th>
      <td>yogurt</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
purchase = pd.read_csv('purchase_history.csv')
purchase.head()
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
      <th>user_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>222087</td>
      <td>27,26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1343649</td>
      <td>6,47,17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>404134</td>
      <td>18,12,23,22,27,43,38,20,35,1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1110200</td>
      <td>9,23,2,20,26,47,37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>224107</td>
      <td>31,18,5,13,1,21,48,16,26,2,44,32,20,37,42,35,4...</td>
    </tr>
  </tbody>
</table>
</div>



#### Data Summary

There are 39,474 purchases made by 24,885 unique customers. Data has no missing values.


```python
item.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 48 entries, 0 to 47
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   Item_name  48 non-null     object
     1   Item_id    48 non-null     int64 
    dtypes: int64(1), object(1)
    memory usage: 896.0+ bytes



```python
purchase.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39474 entries, 0 to 39473
    Data columns (total 2 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   user_id  39474 non-null  int64 
     1   id       39474 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 616.9+ KB



```python
len(purchase.user_id.unique())
```




    24885




```python
purchase.isnull().sum()
```




    user_id    0
    id         0
    dtype: int64



#### Data Pre-processing

In order to get any meaningful insights from the dataset, we will have to some pre-processing. We will create a frequency table for each item by every user. If a user buys a specific item 2 times we will have (user,item) = 2 in the table.


```python
def count_items(df):
    ids = df['id'].str.split(',').sum()
    id_list = [0 for i in range(1,49)]
    for i in ids:
        id_list[int(i)-1]+=1
    return pd.Series(id_list,index = list(range(1,49)))

```


```python
user_item_cnt = purchase.groupby('user_id').apply(count_items)
user_item_cnt.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



### Top 10 customers who bought the most items


```python
user_count = user_item_cnt.sum(axis = 1).sort_values(ascending = False).reset_index().rename(columns = {0:'item_cnt'})
user_count.head(10)
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
      <th>user_id</th>
      <th>item_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>269335</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>367872</td>
      <td>70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>599172</td>
      <td>64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>397623</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>377284</td>
      <td>63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1485538</td>
      <td>62</td>
    </tr>
    <tr>
      <th>6</th>
      <td>917199</td>
      <td>62</td>
    </tr>
    <tr>
      <th>7</th>
      <td>718218</td>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>653800</td>
      <td>60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>828721</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>



### For each item, the customer who bought that product the most


```python
item_max = user_item_cnt.apply(lambda s : pd.Series([s.argmax(),s.max()],index = ['max_user','max_cnt']))
item_max = item_max.transpose()
item_max.index.name = 'Item_id'
item_max
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
      <th>max_user</th>
      <th>max_cnt</th>
    </tr>
    <tr>
      <th>Item_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>512</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>512</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2552</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>92</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3605</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5555</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2926</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2493</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4445</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10238</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6111</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9261</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10799</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2855</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2356</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1175</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6085</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>15193</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>512</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>14623</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>14597</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>19892</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>15251</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3154</td>
      <td>3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1094</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>16026</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>15840</td>
      <td>4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3400</td>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6565</td>
      <td>4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>375</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>4803</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1771</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33</th>
      <td>21668</td>
      <td>3</td>
    </tr>
    <tr>
      <th>34</th>
      <td>5085</td>
      <td>4</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7526</td>
      <td>3</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4445</td>
      <td>4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>749</td>
      <td>4</td>
    </tr>
    <tr>
      <th>38</th>
      <td>4231</td>
      <td>4</td>
    </tr>
    <tr>
      <th>39</th>
      <td>9918</td>
      <td>5</td>
    </tr>
    <tr>
      <th>40</th>
      <td>634</td>
      <td>4</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2192</td>
      <td>4</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1303</td>
      <td>4</td>
    </tr>
    <tr>
      <th>43</th>
      <td>16510</td>
      <td>4</td>
    </tr>
    <tr>
      <th>44</th>
      <td>512</td>
      <td>4</td>
    </tr>
    <tr>
      <th>45</th>
      <td>19857</td>
      <td>5</td>
    </tr>
    <tr>
      <th>46</th>
      <td>20214</td>
      <td>4</td>
    </tr>
    <tr>
      <th>47</th>
      <td>6419</td>
      <td>4</td>
    </tr>
    <tr>
      <th>48</th>
      <td>5575</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



We were able to get all the users who bought maximum number of each item.Identifying a target audience provides a clear focus of whom your business will serve and why those consumers need your goods or services. Determining this information also keeps a target audience at a manageable level.

## Clustering of Items

### Item-Item Similiarity Matrix

To cluster items we would first create a cosine similarity matrix.<br>
***Cosine similarity*** is a metric that can be used to determine how similar the documents/terms are irrespective of their magnitude.
Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space.When plotted on a multi-dimensional space, where each dimension corresponds to a word , the cosine similarity captures the orientation (the angle) of the terms and not the magnitude. If you want the magnitude, compute the Euclidean distance instead.

The cosine similarity is advantageous because even if the two similar items are far apart by the Euclidean distance because of the size (like, the word ‘fruit’ was purchased 10 times by 1 user and 1 time by other) they could still have a smaller angle between them. Smaller the angle, higher the similarity. 


```python
user_item_cnt.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>




```python
user_item_cnt_transpose = user_item_cnt.T
similarity = cosine_similarity(user_item_cnt_transpose,user_item_cnt_transpose)
similarity_df = pd.DataFrame(similarity, index=user_item_cnt.columns, columns=user_item_cnt.columns)
```


```python
similarity_df[:5]
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.506895</td>
      <td>0.420145</td>
      <td>0.296986</td>
      <td>0.271132</td>
      <td>0.388250</td>
      <td>0.271743</td>
      <td>0.335303</td>
      <td>0.403690</td>
      <td>0.390641</td>
      <td>...</td>
      <td>0.388034</td>
      <td>0.390286</td>
      <td>0.358599</td>
      <td>0.393056</td>
      <td>0.395696</td>
      <td>0.396766</td>
      <td>0.390253</td>
      <td>0.394998</td>
      <td>0.392164</td>
      <td>0.328221</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.506895</td>
      <td>1.000000</td>
      <td>0.466874</td>
      <td>0.322744</td>
      <td>0.285125</td>
      <td>0.468199</td>
      <td>0.312200</td>
      <td>0.390521</td>
      <td>0.464872</td>
      <td>0.527894</td>
      <td>...</td>
      <td>0.462968</td>
      <td>0.462548</td>
      <td>0.409401</td>
      <td>0.529100</td>
      <td>0.464579</td>
      <td>0.527325</td>
      <td>0.521058</td>
      <td>0.462407</td>
      <td>0.460257</td>
      <td>0.380077</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.420145</td>
      <td>0.466874</td>
      <td>1.000000</td>
      <td>0.277325</td>
      <td>0.224537</td>
      <td>0.358326</td>
      <td>0.238133</td>
      <td>0.301868</td>
      <td>0.362091</td>
      <td>0.352597</td>
      <td>...</td>
      <td>0.351093</td>
      <td>0.368199</td>
      <td>0.309078</td>
      <td>0.357794</td>
      <td>0.351209</td>
      <td>0.362522</td>
      <td>0.361922</td>
      <td>0.354933</td>
      <td>0.351832</td>
      <td>0.297972</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.296986</td>
      <td>0.322744</td>
      <td>0.277325</td>
      <td>1.000000</td>
      <td>0.162860</td>
      <td>0.247414</td>
      <td>0.166012</td>
      <td>0.216166</td>
      <td>0.252662</td>
      <td>0.258313</td>
      <td>...</td>
      <td>0.245623</td>
      <td>0.261272</td>
      <td>0.210767</td>
      <td>0.244946</td>
      <td>0.253282</td>
      <td>0.253119</td>
      <td>0.250190</td>
      <td>0.253835</td>
      <td>0.260541</td>
      <td>0.218717</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.271132</td>
      <td>0.285125</td>
      <td>0.224537</td>
      <td>0.162860</td>
      <td>1.000000</td>
      <td>0.233618</td>
      <td>0.164699</td>
      <td>0.203245</td>
      <td>0.239445</td>
      <td>0.234254</td>
      <td>...</td>
      <td>0.235543</td>
      <td>0.238557</td>
      <td>0.211006</td>
      <td>0.238466</td>
      <td>0.235064</td>
      <td>0.241835</td>
      <td>0.238087</td>
      <td>0.238247</td>
      <td>0.232387</td>
      <td>0.188269</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



### Feature Selection Using PCA

Now that we have the similarity matrix, we can cluster the items based on the cosine similarity score but before that we will use PCA to reduce the dimension of the dataset.<br>
PCA will help us to discover which dimensions in the dataset best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the explained variance ratio of each dimension — how much variance within the data is explained by that dimension alone.


```python
pca_model = PCA()
items_rotate = pca_model.fit_transform(similarity_df)

items_rotate = pd.DataFrame(items_rotate,index=user_item_cnt.columns,
                        columns=["pc{}".format(idx+1) for idx in range(item.shape[0])])
items_rotate.head()
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
      <th>pc1</th>
      <th>pc2</th>
      <th>pc3</th>
      <th>pc4</th>
      <th>pc5</th>
      <th>pc6</th>
      <th>pc7</th>
      <th>pc8</th>
      <th>pc9</th>
      <th>pc10</th>
      <th>...</th>
      <th>pc39</th>
      <th>pc40</th>
      <th>pc41</th>
      <th>pc42</th>
      <th>pc43</th>
      <th>pc44</th>
      <th>pc45</th>
      <th>pc46</th>
      <th>pc47</th>
      <th>pc48</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.355639</td>
      <td>-0.025103</td>
      <td>-0.051129</td>
      <td>-0.039214</td>
      <td>-0.049529</td>
      <td>0.003300</td>
      <td>0.019830</td>
      <td>-0.011458</td>
      <td>0.084798</td>
      <td>-0.042284</td>
      <td>...</td>
      <td>-0.019760</td>
      <td>-0.012837</td>
      <td>-0.006332</td>
      <td>-0.022289</td>
      <td>-0.002654</td>
      <td>-0.001332</td>
      <td>0.011263</td>
      <td>-0.026911</td>
      <td>-0.129284</td>
      <td>5.732095e-17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.793894</td>
      <td>-0.041103</td>
      <td>0.013244</td>
      <td>-0.009035</td>
      <td>-0.141627</td>
      <td>-0.036540</td>
      <td>-0.114816</td>
      <td>-0.087070</td>
      <td>0.000959</td>
      <td>-0.035030</td>
      <td>...</td>
      <td>-0.039028</td>
      <td>-0.015895</td>
      <td>-0.027280</td>
      <td>0.017406</td>
      <td>0.013680</td>
      <td>0.004437</td>
      <td>-0.030161</td>
      <td>0.006713</td>
      <td>0.379136</td>
      <td>5.732095e-17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.135668</td>
      <td>-0.029584</td>
      <td>-0.028246</td>
      <td>-0.012556</td>
      <td>-0.013317</td>
      <td>0.004220</td>
      <td>-0.020503</td>
      <td>-0.050246</td>
      <td>0.177599</td>
      <td>-0.080864</td>
      <td>...</td>
      <td>0.020455</td>
      <td>-0.007168</td>
      <td>0.040652</td>
      <td>0.004387</td>
      <td>0.029288</td>
      <td>0.000224</td>
      <td>0.009908</td>
      <td>0.001288</td>
      <td>-0.035429</td>
      <td>5.732095e-17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.532034</td>
      <td>0.020259</td>
      <td>-0.022628</td>
      <td>-0.037650</td>
      <td>0.022756</td>
      <td>-0.046219</td>
      <td>0.042124</td>
      <td>-0.132828</td>
      <td>0.651482</td>
      <td>-0.274771</td>
      <td>...</td>
      <td>0.000133</td>
      <td>0.002444</td>
      <td>0.005894</td>
      <td>-0.003617</td>
      <td>-0.005383</td>
      <td>0.001555</td>
      <td>-0.006114</td>
      <td>0.007527</td>
      <td>0.020217</td>
      <td>5.732095e-17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.678770</td>
      <td>-0.093874</td>
      <td>-0.347623</td>
      <td>-0.232866</td>
      <td>-0.090366</td>
      <td>-0.057124</td>
      <td>-0.018074</td>
      <td>0.038866</td>
      <td>-0.048059</td>
      <td>-0.014378</td>
      <td>...</td>
      <td>-0.008466</td>
      <td>0.011890</td>
      <td>0.002598</td>
      <td>0.007605</td>
      <td>0.009321</td>
      <td>-0.003278</td>
      <td>0.006540</td>
      <td>0.000677</td>
      <td>0.023603</td>
      <td>5.732095e-17</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



### Dimensionality Reduction
When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the cumulative explained variance ratio is extremely important for knowing how many dimensions are necessary for the problem. We can even visualize the components and variance explained by each of them.


```python
fig = plt.figure(figsize=(10, 10))
plt.bar(range(1,len(pca_model.explained_variance_)+1),pca_model.explained_variance_)
plt.ylabel('Explained variance')
plt.xlabel('Components')
plt.title("Explained variance plot")
```




    Text(0.5, 1.0, 'Explained variance plot')




![png](/images/clustering/output_27_1.png)



```python
plt.plot(pca_model.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Explained Variance Ratio Plot")
plt.show()
```


![png](/images/clustering/output_28_0.png)



```python
# show the total variance which can be explained by first K principle components
explained_variance_by_k = pca_model.explained_variance_ratio_.cumsum()

plt.plot(range(1,len(explained_variance_by_k)+1),explained_variance_by_k,marker="*")
plt.title("PCA Cumulative Variance Plot")
```




    Text(0.5, 1.0, 'PCA Cumulative Variance Plot')




![png](/images/clustering/output_29_1.png)


We can see 35 principal components can explain upto 90% variance in the dataset. 


```python
sum(pca_model.explained_variance_ratio_[:35])
```




    0.9036104968698064



### K-means Clustering

We are going to use K-means clustering to identify the clusters for our items.<br>
Advantages of k-means
1. Relatively simple to implement.
2. Scales to large data sets.
3. Guarantees convergence.
4. Easily adapts to new examples.
5. Generalizes to clusters of different shapes and sizes, such as elliptical clusters.<br>

***Random Initialization***:
In some cases, if the initialization of clusters is not appropriate, K-Means can result in arbitrarily bad clusters. This is where K-Means++ helps. It specifies a procedure to initialize the cluster centers before moving forward with the standard k-means clustering algorithm. It will randomly choose a centroid, calculate the distance of every point from it and the point which has the maximum distance from the centroid will be chosen as the second centroid.

#### Creating Clusters
When the number of clusters is not known a priori like in this problem, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's ***silhouette coefficient*** and ***inertia***.<br>
The silhouette coefficient for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the mean silhouette coefficient provides for a simple scoring method of a given clustering.<br>
Inertia: It is defined as the mean squared distance between each instance and its closest centroid. Logically, as per the definition lower the inertia better the model.


```python
#We will use 25 pca components
items_filter = items_rotate.values[:,:25]
```


```python
clusters = range(5, 20)
inertias = []
silhouettes = []

for n_clusters in clusters:
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_jobs=1)
    label = model.fit_predict(items_filter)
    
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(items_filter, label))
```


```python
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
ax[0].plot(clusters, inertias, 'o-', label='Sum of Squared Distances')
ax[0].grid(True)
ax[1].plot(clusters, silhouettes, 'o-', label='Silhouette Coefficient')
ax[1].grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

```


![png](/images/clustering/output_35_0.png)


Silhouette coefficient plateaued after 14 so 14 clusters look optimal.

### Cluster Visualization

We can see 14 clusters are created and items are clearly sorted. We were able to reduce dimensionality and ended up using just 25 features.<br>
Items are sorted as bakery items(Cluster 4) such as bagels
,tortillas
,dinner rolls
,sandwich loaves and canned goods(cluster 1) such as  spaghetti sauce
,canned vegetables
,ketchup. Similarly, we have other clusters as well.


```python
def show_clusters(items_rotated,labels):
    """
    plot and print clustering result
    """
    fig = plt.figure(figsize=(15, 15))
    colors =  itertools.cycle (["blue","green","red","cyan","yellow","purple","orange","olive","seagreen","magenta","brown","lightpink","indigo","aqua","teal"])

    grps = items_rotated.groupby(labels)
    for label,grp in grps:
        #print(label,grp)
        plt.scatter(grp.pc1,grp.pc2,c=next(colors),label = label)

        print("*********** Label [{}] ***********".format(label))
        print(grp.index)
        for j in grp.index:
            
            names = item[item['Item_id'] == j]["Item_name"].to_string(index = False)
            print(j,names)
            
        

    # annotate
    for itemid in items_rotated.index:
        x = items_rotated.loc[itemid,"pc1"]
        y = items_rotated.loc[itemid,"pc2"]
        name = item[item['Item_id'] == itemid]["Item_name"].to_string(index = False)
        plt.text(x,y,name)
```


```python
def cluster(n_clusters,n_components=48):
    """
    n_components=K, means use first K principle components in the clustering
    n_clusters: the number of clusters we want to cluster
    """
    print("first {} PC explain {:.1f}% variances".format(n_components,
                                                         100 * sum(pca_model.explained_variance_ratio_[:n_components])))

    kmeans_model = KMeans(n_clusters=n_clusters,init='k-means++', random_state=42, n_jobs=1)
    kmeans_model.fit(items_rotate.values[:, :n_components])

    # display results
    show_clusters(items_rotate, kmeans_model.labels_)
```


```python

cluster(n_clusters = 14,n_components=25)
```

    first 25 PC explain 79.8% variances
    *********** Label [0] ***********
    Int64Index([26, 28, 41], dtype='int64')
    26  spaghetti sauce
    28  canned vegetables
    41  ketchup
    *********** Label [1] ***********
    Int64Index([30], dtype='int64')
    30  flour
    *********** Label [2] ***********
    Int64Index([1, 3, 20, 25, 32, 40, 46], dtype='int64')
    1  sugar
    3  pet items
    20  grapefruit
    25  cherries
    32  apples
    40  berries
    46  bananas
    *********** Label [3] ***********
    Int64Index([12, 19, 29, 35], dtype='int64')
    12  shampoo
    19  shaving cream
    29  hand soap
    35  soap
    *********** Label [4] ***********
    Int64Index([13, 34, 37, 39], dtype='int64')
    13  bagels
    34  tortillas
    37  dinner rolls
    39  sandwich loaves
    *********** Label [5] ***********
    Int64Index([7, 15, 24, 33], dtype='int64')
    7  sandwich bags
    15  aluminum foil
    24  paper towels
    33  toilet paper
    *********** Label [6] ***********
    Int64Index([2, 10, 42, 44, 45], dtype='int64')
    2  lettuce
    10  carrots
    42  cucumbers
    44  broccoli
    45  cauliflower
    *********** Label [7] ***********
    Int64Index([5, 11, 31], dtype='int64')
    5  waffles
    11  cereals
    31  pasta
    *********** Label [8] ***********
    Int64Index([8, 14, 16, 21, 48], dtype='int64')
    8  butter
    14  eggs
    16  milk
    21  cheeses
    48  yogurt
    *********** Label [9] ***********
    Int64Index([9, 23, 38, 43], dtype='int64')
    9  soda
    23  tea
    38  juice
    43  coffee
    *********** Label [10] ***********
    Int64Index([22, 36], dtype='int64')
    22  frozen vegetables
    36  ice cream
    *********** Label [11] ***********
    Int64Index([18, 27], dtype='int64')
    18  laundry detergent
    27  dishwashing 
    *********** Label [12] ***********
    Int64Index([4], dtype='int64')
    4  baby items
    *********** Label [13] ***********
    Int64Index([6, 17, 47], dtype='int64')
    6  poultry
    17  beef
    47  pork



![png](/images/clustering/output_40_1.png)



```python

```
