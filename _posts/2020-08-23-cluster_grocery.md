---
layout: post
title:  "Clustering Grocery Items"
date:   2020-08-23 15:03:14 -0400
categories: jekyll update
---


Company XYZ is an online grocery store. In the current version of the website, they have manually grouped the items into a few categories based on their experience.

However, they now have a lot of data about user purchase history. Therefore, they would like to put the data into use! This is what they asked you to do:

The company founder wants to meet with some of the best customers to go through a focus group with them. You are asked to send the ID of the following customers to the founder:
the customer who bought the most items overall in her lifetime
for each item, the customer who bought that product the most
Cluster items based on user co-purchase history. That is, create clusters of products that have the highest probability of being bought together. The goal of this is to replace the old/manually created categories with these new ones. Each item can belong to just one cluster.


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

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
```


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




```python
item.describe()
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
      <th>Item_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>48.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.50</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.75</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.50</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>36.25</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48.00</td>
    </tr>
  </tbody>
</table>
</div>




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




```python
sorted(item['Item_id'].unique())
```




    [1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28,
     29,
     30,
     31,
     32,
     33,
     34,
     35,
     36,
     37,
     38,
     39,
     40,
     41,
     42,
     43,
     44,
     45,
     46,
     47,
     48]




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



# The customer who bought the most items overall in her lifetime


```python
user_count = user_item_cnt.sum(axis = 1).sort_values(ascending = False).reset_index().rename(columns = {0:'item_cnt'})
user_count.head()
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
  </tbody>
</table>
</div>



# For each item, the customer who bought that product the most


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




```python
pd.merge(left = item,right = item_max,on ='Item_id',how = 'left' )
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
      <th>max_user</th>
      <th>max_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>coffee</td>
      <td>43</td>
      <td>16510</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tea</td>
      <td>23</td>
      <td>15251</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>juice</td>
      <td>38</td>
      <td>4231</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>soda</td>
      <td>9</td>
      <td>4445</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sandwich loaves</td>
      <td>39</td>
      <td>9918</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dinner rolls</td>
      <td>37</td>
      <td>749</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tortillas</td>
      <td>34</td>
      <td>5085</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bagels</td>
      <td>13</td>
      <td>10799</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>canned vegetables</td>
      <td>28</td>
      <td>3400</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>spaghetti sauce</td>
      <td>26</td>
      <td>16026</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ketchup</td>
      <td>41</td>
      <td>2192</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cheeses</td>
      <td>21</td>
      <td>14597</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>eggs</td>
      <td>14</td>
      <td>2855</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>milk</td>
      <td>16</td>
      <td>1175</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>yogurt</td>
      <td>48</td>
      <td>5575</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>butter</td>
      <td>8</td>
      <td>2493</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cereals</td>
      <td>11</td>
      <td>6111</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>flour</td>
      <td>30</td>
      <td>375</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sugar</td>
      <td>1</td>
      <td>512</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>pasta</td>
      <td>31</td>
      <td>4803</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>waffles</td>
      <td>5</td>
      <td>3605</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>frozen vegetables</td>
      <td>22</td>
      <td>19892</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ice cream</td>
      <td>36</td>
      <td>4445</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>poultry</td>
      <td>6</td>
      <td>5555</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>beef</td>
      <td>17</td>
      <td>6085</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>pork</td>
      <td>47</td>
      <td>6419</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>bananas</td>
      <td>46</td>
      <td>20214</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>berries</td>
      <td>40</td>
      <td>634</td>
      <td>4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cherries</td>
      <td>25</td>
      <td>1094</td>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>grapefruit</td>
      <td>20</td>
      <td>14623</td>
      <td>4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>apples</td>
      <td>32</td>
      <td>1771</td>
      <td>4</td>
    </tr>
    <tr>
      <th>31</th>
      <td>broccoli</td>
      <td>44</td>
      <td>512</td>
      <td>4</td>
    </tr>
    <tr>
      <th>32</th>
      <td>carrots</td>
      <td>10</td>
      <td>10238</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33</th>
      <td>cauliflower</td>
      <td>45</td>
      <td>19857</td>
      <td>5</td>
    </tr>
    <tr>
      <th>34</th>
      <td>cucumbers</td>
      <td>42</td>
      <td>1303</td>
      <td>4</td>
    </tr>
    <tr>
      <th>35</th>
      <td>lettuce</td>
      <td>2</td>
      <td>512</td>
      <td>5</td>
    </tr>
    <tr>
      <th>36</th>
      <td>laundry detergent</td>
      <td>18</td>
      <td>15193</td>
      <td>5</td>
    </tr>
    <tr>
      <th>37</th>
      <td>dishwashing</td>
      <td>27</td>
      <td>15840</td>
      <td>4</td>
    </tr>
    <tr>
      <th>38</th>
      <td>paper towels</td>
      <td>24</td>
      <td>3154</td>
      <td>3</td>
    </tr>
    <tr>
      <th>39</th>
      <td>toilet paper</td>
      <td>33</td>
      <td>21668</td>
      <td>3</td>
    </tr>
    <tr>
      <th>40</th>
      <td>aluminum foil</td>
      <td>15</td>
      <td>2356</td>
      <td>3</td>
    </tr>
    <tr>
      <th>41</th>
      <td>sandwich bags</td>
      <td>7</td>
      <td>2926</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>shampoo</td>
      <td>12</td>
      <td>9261</td>
      <td>3</td>
    </tr>
    <tr>
      <th>43</th>
      <td>soap</td>
      <td>35</td>
      <td>7526</td>
      <td>3</td>
    </tr>
    <tr>
      <th>44</th>
      <td>hand soap</td>
      <td>29</td>
      <td>6565</td>
      <td>4</td>
    </tr>
    <tr>
      <th>45</th>
      <td>shaving cream</td>
      <td>19</td>
      <td>512</td>
      <td>3</td>
    </tr>
    <tr>
      <th>46</th>
      <td>baby items</td>
      <td>4</td>
      <td>92</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>pet items</td>
      <td>3</td>
      <td>2552</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



# Item-Item Similiarity Matrix


```python
from sklearn.preprocessing import normalize
user_item_norm = normalize(user_item_cnt.values, axis=0)
similarity = np.dot(user_item_norm.T,user_item_norm)  # calculate the similarity matrix
similarity_df = pd.DataFrame(similarity, index=user_item_cnt.columns, columns=user_item_cnt.columns)
similarity_df.head()
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
      <th>Item_id</th>
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
      <th>Item_id</th>
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




```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

# Clustering(K-means with PCA)


```python
pca = PCA()
items_rotate = pca.fit_transform(similarity_df)

items_rotate
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
    <tr>
      <th>Item_id</th>
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




```python
# show the total variance which can be explained by first K principle components
explained_variance_by_k = pca.explained_variance_ratio_.cumsum()
explained_variance_by_k
plt.plot(range(1,len(explained_variance_by_k)+1),explained_variance_by_k,marker="*")
```




    [<matplotlib.lines.Line2D at 0x11f761b80>]




![png](output_22_1.png)



```python
sum(pca.explained_variance_ratio_[:30])
```




    0.8568104305151689




```python
def show_clusters(items_rotated,labels):
    """
    plot and print clustering result
    """
    fig = plt.figure(figsize=(15, 15))
    colors =  itertools.cycle (["b","g","r","c","m","y","k"])

    grps = items_rotated.groupby(labels)
    for label,grp in grps:
        #print(label,grp)
        plt.scatter(grp.pc1,grp.pc2,c=next(colors),label = label)

        print("*********** Label [{}] ***********".format(label))
        print(grp.index)
        for j in grp.index:
            
            names = item[item['Item_id'] == j]["Item_name"].to_string(index = False)
            print(j,names)
            
        #names = item.loc[[grp.index],"Item_name"]
        

    # annotate
    for itemid in items_rotated.index:
        x = items_rotated.loc[itemid,"pc1"]
        y = items_rotated.loc[itemid,"pc2"]
        name = item[item['Item_id'] == itemid]["Item_name"].to_string(index = False)
        #name = re.sub('\W', ' ', name)
        plt.text(x,y,name)
```


```python
def cluster(n_clusters,n_components=48):
    """
    n_components=K, means use first K principle components in the clustering
    n_clusters: the number of clusters we want to cluster
    """
    print("first {} PC explain {:.1f}% variances".format(n_components,
                                                         100 * sum(pca.explained_variance_ratio_[:n_components])))

    kmeans = KMeans(n_clusters=n_clusters,init='k-means++', random_state=42, n_jobs=1)
    kmeans.fit(items_rotate.values[:, :n_components])

    # display results
    show_clusters(items_rotate, kmeans.labels_)
```


```python
kmeans = KMeans(n_clusters=5)
kmeans.fit(items_rotate.values[:, :30])
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)




```python
cluster(n_clusters = 16,n_components=30)
```

    first 30 PC explain 85.7% variances
    *********** Label [0] ***********
    Int64Index([6, 17, 47], dtype='int64', name='Item_id')
    6  poultry
    17  beef
    47  pork
    *********** Label [1] ***********
    Int64Index([30], dtype='int64', name='Item_id')
    30  flour
    *********** Label [2] ***********
    Int64Index([1, 3, 20, 22, 25, 32, 40, 46], dtype='int64', name='Item_id')
    1  sugar
    3  pet items
    20  grapefruit
    22  frozen vegetables
    25  cherries
    32  apples
    40  berries
    46  bananas
    *********** Label [3] ***********
    Int64Index([4, 19], dtype='int64', name='Item_id')
    4  baby items
    19  shaving cream
    *********** Label [4] ***********
    Int64Index([24, 33], dtype='int64', name='Item_id')
    24  paper towels
    33  toilet paper
    *********** Label [5] ***********
    Int64Index([5, 11], dtype='int64', name='Item_id')
    5  waffles
    11  cereals
    *********** Label [6] ***********
    Int64Index([12, 29, 35], dtype='int64', name='Item_id')
    12  shampoo
    29  hand soap
    35  soap
    *********** Label [7] ***********
    Int64Index([13, 34, 37, 39], dtype='int64', name='Item_id')
    13  bagels
    34  tortillas
    37  dinner rolls
    39  sandwich loaves
    *********** Label [8] ***********
    Int64Index([8, 14, 16, 21, 48], dtype='int64', name='Item_id')
    8  butter
    14  eggs
    16  milk
    21  cheeses
    48  yogurt
    *********** Label [9] ***********
    Int64Index([31], dtype='int64', name='Item_id')
    31  pasta
    *********** Label [10] ***********
    Int64Index([2, 10, 42, 44, 45], dtype='int64', name='Item_id')
    2  lettuce
    10  carrots
    42  cucumbers
    44  broccoli
    45  cauliflower
    *********** Label [11] ***********
    Int64Index([9, 23, 36, 38, 43], dtype='int64', name='Item_id')
    9  soda
    23  tea
    36  ice cream
    38  juice
    43  coffee
    *********** Label [12] ***********
    Int64Index([7], dtype='int64', name='Item_id')
    7  sandwich bags
    *********** Label [13] ***********
    Int64Index([15], dtype='int64', name='Item_id')
    15  aluminum foil
    *********** Label [14] ***********
    Int64Index([26, 28, 41], dtype='int64', name='Item_id')
    26  spaghetti sauce
    28  canned vegetables
    41  ketchup
    *********** Label [15] ***********
    Int64Index([18, 27], dtype='int64', name='Item_id')
    18  laundry detergent
    27  dishwashing 



![png](output_27_1.png)



```python
items_filter = items_rotate.values[:,:30]
```


```python
# determine the best number of clusters
clusters = range(2, 30)
inertias = []
silhouettes = []

for n_clusters in clusters:
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_jobs=1)
    kmeans = kmeans.fit(items_filter)
    label = kmeans.predict(items_filter)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(items_filter, label))
```


```python
# visualization
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
ax[0].plot(clusters, inertias, 'o-', label='Sum of Squared Distances')
ax[0].grid(True)
ax[1].plot(clusters, silhouettes, 'o-', label='Silhouette Coefficient')
ax[1].grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

#16 clusters seem optimal based on silhouette score. Higher the better.
```


![png](output_30_0.png)



```python

```
