# -*- coding: utf-8 -*- 

# For data manipulation
import pandas as pd         

# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# ## 1.2 Load data from the CSV files
# Instacart provides 6 CSV files, which we have to load into Python. Towards this end, we use the .read_csv() function, which is included in the Pandas package. Reading in data with the .read_csv( ) function returns a DataFrame.
# 
# First we connect to the Kaggle API in order to download the zip file with the 6 CSVs. Then we unzip it in a new folder named input, and we unzip all the zip files.


# connect to kaggle api and download files (zip)
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi({"username":"charatrypologou","key":"9bab08071e50040cb77728875d96e02c"})
api.authenticate()
files = api.competition_download_files("Instacart-Market-Basket-Analysis")


# In[ ]:


import zipfile
with zipfile.ZipFile('Instacart-Market-Basket-Analysis.zip', 'r') as zip_ref:
    zip_ref.extractall('./input')


# In[ ]:


import os
working_directory = os.getcwd()+'/input'
os.chdir(working_directory)
for file in os.listdir(working_directory):   # get the list of files
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
           item.extractall()  # extract it in the working directory


# In[ ]:


orders = pd.read_csv('../input/orders.csv' )
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')


# This step results in the following DataFrames:
# * <b>orders</b>: This table includes all orders, namely prior, train, and test. It has single primary key (<b>order_id</b>).
# * <b>order_products_train</b>: This table includes training orders. It has a composite primary key (<b>order_id and product_id</b>) and indicates whether a product in an order is a reorder or not (through the reordered variable).
# * <b>order_products_prior </b>: This table includes prior orders. It has a composite primary key (<b>order_id and product_id</b>) and indicates whether a product in an order is a reorder or not (through the reordered variable).
# * <b>products</b>: This table includes all products. It has a single primary key (<b>product_id</b>)
# * <b>aisles</b>: This table includes all aisles. It has a single primary key (<b>aisle_id</b>)
# * <b>departments</b>: This table includes all departments. It has a single primary key (<b>department_id</b>)

# If you want to reduce the execution time of this Kernel you can use the following piece of code by uncomment it. This will trim the orders DataFrame and will keep a 10% random sample of the users. You can use this for experimentation.

# In[ ]:



#### Remove triple quotes to trim your dataset and experiment with your data
### COMMANDS FOR CODING TESTING - Get 10% of users 
###orders = orders.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.1, random_state=25))] 


# We now use the .head( ) method in order to visualise the first 10 rows of these tables. Click the Output button below to see the tables.

# In[ ]:


orders.head()


# In[ ]:


order_products_train.head()


# In[ ]:


order_products_prior.head()


# In[ ]:


products.head()


# In[ ]:


aisles.head()


# In[ ]:


departments.head()


# ## 1.3 Reshape data
# We transform the data in order to facilitate their further analysis. First, we convert character variables into categories so we can use them in the creation of the model. In Python, a categorical variable is called category and has a fixed number of different values.

# In[6]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# ## 1.4 Create a DataFrame with the orders and the products that have been purchased on prior orders (op)
# We create a new DataFrame, named <b>op</b> which combines (merges) the DataFrames <b>orders</b> and <b>order_products_prior</b>. Bear in mind that <b>order_products_prior</b> DataFrame includes only prior orders, so the new DataFrame <b>op</b>  will contain only these observations as well. Towards this end, we use pandas' merge function with how='inner' argument, which returns records that have matching values in both DataFrames. 
# <img src="https://i.imgur.com/zEK7FpY.jpg" width="400">

# In[ ]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


## First approach in one step:
# Create distinct groups for each user, identify the highest order number in each group, save the new column to a DataFrame
user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
user.head()

## Second approach in two steps: 
#1. Save the result as DataFrame with Double brackets --> [[ ]] 
#user = op.groupby('user_id')[['order_number']].max()
#2. Rename the label of the column
#user.columns = ['u_total_orders']
#user.head()


# In[ ]:


# Reset the index of the DF so to bring user_id from index to column (pre-requisite for step 2.4)
user = user.reset_index()
user.head()


# ### 2.1.2 How frequent a customer has reordered products
# 
# This feature is a ratio which shows for each user in what extent has products that have been reordered in the past: <br>
# So we create the following ratio: <br>
# 
# <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\large&space;probability\&space;reordered\&space;(user\_id)=&space;\frac{total\&space;times\&space;of\&space;reorders}{total\&space;number\&space;of\&space;purchased\&space;products\&space;from\&space;all\&space;baskets}" title="probability\ reordered\ (user\_id)= \frac{total\ times\ of\ reorders}{total\ number\ of\ purchased\ products\ from\ all\ baskets}" />
# 
# The nominator is a counter for all the times a user has reordered products (value on reordered=1), the denominator is a counter of all the products that have been purchased on all user's orders (reordered=0 & reordered=1).
# 
# E.g., for a user that has ordered 6 products in total, where 3 times were reorders, the ratio will be:
# 
# ![example ratio](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\large&space;&space;mean=&space;\frac{0&plus;1&plus;0&plus;0&plus;1&plus;1}{6}&space;=&space;0,5) 
# 
# To create the above ratio we .groupby() order_products_prior by each user and then calculate the mean of reordered.
# 

# In[ ]:


u_reorder = op.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio')
u_reorder = u_reorder.reset_index()
u_reorder.head()


# The new feature will be merged with the user DataFrame (section 2.1.1) which keep all the features based on users. We perform a left join as we want to keep all the users that we have created on the user DataFrame
# 
# <img src="https://i.imgur.com/wMmC4hb.jpg" width="400">

# In[ ]:
#First, we will keep the 5 last orders of all users

#Keep last 5 orders
op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1 
op.head()

#Keep last 5 orders
op5 = op[op.order_number_back <= 5]
op5.head()

#Last 5 oders of each user and product_id
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five.head()

#Size of orders for each user
u_last_five = last_five.groupby ('user_id')[["times_last5"]].count()
u_last_five.columns = ['size_last5']
u_last_five.head()

#Mean of last 5 orders of each customer
#u_last_five ['mean_size_last5']= u_last_five.size_last5 / 5
#u_last_five.head()

#Max days of 5 last orders for each user
#u_last_five ['max_days_5'] = op5.groupby ('user_id') [["days_since_prior_order"]].max()
#u_last_five.head()

#Mean days of 5 last orders for each user
#u_last_five ['mean_days_5'] = op5.groupby ('user_id') [["days_since_prior_order"]].mean()
#u_last_five.head()

#Merge user with u_reorder
user = user.merge(u_reorder, on='user_id', how='left')
del u_reorder
gc.collect()
user.head()

#Merge user with last_five 
user = user.merge (u_last_five, on = "user_id" , how ="left")
del u_last_five
gc.collect()
user.head()

#Days of week and hour of the day of the last 5 orders for each user
#user_day_hour = pd.merge(user, op5[['user_id', 'order_dow', 'order_hour_of_day']], on='user_id', how='left')
#user_day_hour.head()

#user = user.merge (user_day_hour, on = "user_id" , how ="left")
#del user_day_hour
#gc.collect()
#user.head()
# ## 2.2 Create product predictors
# We create the following predictors:
# - 2.2.1 Number of purchases for each product
# - 2.2.2 What is the probability for a product to be reordered
# 
# ### 2.2.1 Number of purchases for each product
# We calculate the total number of purchases for each product (from all customers). We create a **prd** DataFrame to store the results.

# In[ ]:


# Create distinct groups for each product, count the orders, save the result for each product to a new DataFrame  
prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd.head()


# In[ ]:


# Reset the index of the DF so to bring product_id rom index to column (pre-requisite for step 2.4)
prd = prd.reset_index()
prd.head()


# ## 2.2.2 What is the probability for a product to be reordered
# In this section we want to find the products which have the highest probability of being reordered. Towards this end it is necessary to define the probability as below:
# <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\large&space;probability\&space;reordered\&space;(product\_id)=&space;\frac{number\&space;of\&space;reorders}{total\&space;number\&space;of\&space;orders\&space;}" title="probability\ reordered\ (product\_id)= \frac{number\ of\ reorders}{total\ number\ of\ orders\ }" />
# 
# Example: The product with product_id=2 is included in 90 purchases but only 12 are reorders. So we have:  
# 
# <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\large&space;p\_reorder\(product\_id\mathop{==}&space;2&space;)=&space;\frac{12}{90}=&space;0,133" title="\large p\_reorder\(product\_id\mathop{==} 2 )= \frac{12}{90}= 0,133" />
# 
# ### 2.2.2.1 Remove products with less than 40 purchases - Filter with .shape[0]
# Before we proceed to this estimation, we remove all these products that have less than 40 purchases in order the calculation of the aforementioned ratio to be meaningful.
# 
# Using .groupby() we create groups for each product and using .filter( ) we keep only groups with more than 40 rows. Towards this end, we indicate a lambda function.

# In[ ]:


# execution time: 25 sec
# the x on lambda function is a temporary variable which represents each group
# shape[0] on a DataFrame returns the number of rows
p_reorder = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
p_reorder.head()


# ### 2.2.2.2 Group products, calculate the mean of reorders
# 
# To calculate the reorder probability we will use the aggregation function mean() to the reordered column. In the reorder data frame, the reordered column indicates that a product has been reordered when the value is 1.
# 
# The .mean() calculates how many times a product has been reordered, divided by how many times has been ordered in total. 
# 
# E.g., for a product that has been ordered 9 times in total, where 4 times has been reordered, the ratio will be:
# 
# ![example ratio](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\large&space;&space;mean=&space;\frac{0&plus;1&plus;0&plus;0&plus;1&plus;1&plus;0&plus;0&plus;1}{9}&space;=&space;0,44) 
# 
# We calculate the ratio for each product. The aggregation function is limited to column 'reordered' and it calculates the mean value of each group.

# In[ ]:

p_reorder = p_reorder.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')
p_reorder = p_reorder.reset_index()
p_reorder.head()

#Mean position of each product
p_reorder['avg_position'] = op.groupby ('product_id')[["add_to_cart_order"]].mean()
p_reorder.head()

#Probability of reordered for each product within last 5 orders of the users
op5_ratio = op5.groupby ('product_id')["reordered"].mean().to_frame ('p_reorder_last5')
op5_ratio = op5_ratio.reset_index()
op5_ratio.head()

# ### 2.2.2.3 Merge the new feature on prd DataFrame
# The new feature will be merged with the prd DataFrame (section 2.2.1) which keep all the features based on products. We perform a left join as we want to keep all the products that we have created on the prd DataFrame
# <img src="https://i.imgur.com/dOVWPKb.jpg" width="400">

# In[ ]:

#Merge the prd DataFrame with reorder
prd = prd.merge(p_reorder, on='product_id', how='left')

#delete the reorder DataFrame
del p_reorder
gc.collect()
prd.head()

#Merge the prd DataFrame with op5
prd = prd.merge(op5_ratio, on='product_id', how='left')
gc.collect()
prd.head()

# #### 2.2.2.4 Fill NaN values
# As you may notice, there are product with NaN values. This regards the products that have been purchased less than 40 times from all users and were not included in the p_reorder DataFrame. **As we performed a left join with prd DataFrame, all the rows with products that had less than 40 purchases from all users, will get a NaN value.**
# 
# For these products we their NaN value with zero (0):

# In[ ]:


prd['p_reorder_ratio'] = prd['p_reorder_ratio'].fillna(value=0)
prd['avg_position'] = prd ['avg_position'].fillna(value=0)
prd['p_reorder_last5'] = prd['p_reorder_last5'].fillna(value=0)
prd.head()


# > Our final DataFrame should not have any NaN values, otherwise the fitting process (chapter 4) will throw an error!

# ## 2.3 Create user-product predictors
# We create the following predictors:
# - 2.3.1 How many times a user bought a product
# - 2.3.2 How frequently a customer bought a product after its first purchase
# - 2.3.3 How many times a customer bought a product on its last 5 orders
# 
# ### 2.3.1 How many times a user bought a product
# We create different groups that contain all the rows for each combination of user and product. With the aggregation function .count( ) we get how many times each user bought a product. We save the results on new **uxp** DataFrame.

# In[ ]:


# Create distinct groups for each combination of user and product, count orders, save the result for each user X product to a new DataFrame 
uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
uxp.head()

# In[ ]:


# Reset the index of the DF so to bring user_id & product_id rom indices to columns (pre-requisite for step 2.4)
uxp = uxp.reset_index()
uxp.head()

uxp_last5 = op5.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought_last5')
uxp_last5.head()

uxp_last5 = uxp_last5.reset_index()
uxp_last5 = uxp_last5.fillna(0)
uxp_last5.head()

uxp = uxp.merge (uxp_last5 , on=['user_id', 'product_id'], how='left')
uxp.head()
del uxp_last5
# ### 2.3.2 How frequently a customer bought a product after its first purchase
# This ratio is a metric that describes how many times a user bought a product out of how many times she had the chance to a buy it (starting from her first purchase of the product):
# 
# <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\large&space;probability\&space;reordered\&space;(user\_id\&space;,&space;product\_id)&space;=&space;\frac{Times\_Bought\_N}{Order\_Range\_D}" title="\large probability\ reordered\ (user\_id\ , product\_id) = \frac{Times\_Bought\_N}{Order\_Range\_D}" />
# 
# * Times_Bought_N = Times a user bought a product
# * Order_Range_D = Total orders placed since the first user's order of a product
# 
# To clarify this, we examine the use with user_id:1 and the product with product_id:13032. User 1 has made 10 orders in total.She has bought the product 13032 **for first time in her 2nd order** and she has bought the same product 3 times in total. The user was able to buy the product 9 times (starting from her 2nd order until her last order). As a result, she has bought it 3 out of 9 times, meaning reorder_ratio=3/9= 0,333.
# 
# The Order_Range_D variable is created using two supportive variables:
# * Total_orders = Total number of orders of each user
# * First_order_number = The order number where the customer bought a product for first time
# 
# In the next blocks we show how we create:
# 1. The numerator 'Times_Bought_N'
# 2. The denumerator 'Order_Range_D' with the use of the supportive variables 'total_orders' & 'first_order_number' 
# 3. Our final ratio 'uxp_order_ratio'

# ### 2.3.2.1 Calculate the numerator ('Times_Bought_N')
# 
# To answer this question we simply .groupby( ) user_id & product_id and we count the instances of order_id for each group.

# In[ ]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# ### 2.3.2.2 Calculate the denumerator ('Order_Range_D')
# To calculate the denumerator, we first calculate the total orders of each user & first order number for each user and every product purchase.
# 
# In order to calculate the total number of orders of each cutomer ('total_orders') we .groupby( ) only by the user_id, we keep the column order_number and we get its highest value with the aggregation function .mean()

# In[ ]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# In order to calculate the order number where the user bought a product for first time ('first_order_number') we .groupby( ) by both user_id & product_id and we select the order_number column and we retrieve the .min( ) value.

# In[ ]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# We merge the first order number with the total_orders DataFrame. As total_orders refers to all users, where first_order_no refers to unique combinations of user & product, we perform a right join:
# <img src="https://i.imgur.com/bhln0tn.jpg" width="250">

# In[ ]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# The denominator ('Order_Range_D') now can be created with simple operations between the columns of span DataFrame:

# In[ ]:


# The +1 includes in the difference the first order were the product has been purchased
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# ### 2.3.2.3 Create the final ratio "uxp_reorder_ratio"
# 
# We merge **times** DataFrame, which contains the numerator, and **span** DataFrame, which contains the denumerator of our desired ratio. **As both variables derived from the combination of users & products, any type of join will keep all the combinations.**
# 
# <img src="https://i.imgur.com/h7m1bFh.jpg" width="250">

# In[ ]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# 
# We divide the Times_Bought_N by the Order_Range_D for each user and product.

# In[ ]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# 
# We select to keep only the 'user_id', 'product_id' and the final feature 'uxp_reorder_ratio'
# 

# In[ ]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[ ]:


#Remove temporary DataFrames
del [times, first_order_no, span]

#Find one-shot ratio (1)
#item = op.groupby (['product_id', 'user_id'])[['order_id']].count()
#item.columns = ['total']
#item.head()

#Find one-shot ratio (2)
#item_one = item [item.total==1]
#item_one.head()

#Find one-shot ratio (3)
#item_one = item_one.groupby('product_id')[['total']].count()
#item_one.columns = ['customers_one_shot']        
#item_one.head()

#item = item.reset_index(1)
#item.head()

#Find unique customers of each product
#item_size = item.groupby('product_id')[['user_id']].count()
#item_size.columns = ['unique_customers']
#item_size.head()

#Merge the results
#results = pd.merge (item_one, item_size, on = 'product_id', how = 'right')
#results['one_shot_ratio'] = results ['customers_one_shot']/results['unique_customers']
#results.head()

#Merge uxp_ratio with the results
#uxp_ratio = uxp_ratio.merge (results, on = 'product_id', how = 'left')
#uxp_ratio.head()

#Fill NaN values
#uxp_ratio['one_shot_ratio'] = uxp_ratio['one_shot_ratio'].fillna(0)
#uxp_ratio.head()

#Delete 
#del [item, item_size, item_one, results]
#uxp_ratio = uxp_ratio.drop(['customers_one_shot', 'unique_customers'], axis=1)
#uxp_ratio.head()

# ### 2.3.2.4 Merge the final feature with uxp DataFrame
# The new feature will be merged with the uxp DataFrame (section 2.3.1) which keep all the features based on combinations of user-products. We perform a left join as we want to keep all the user-products that we have created on the uxp DataFrame
# 
# <img src="https://i.imgur.com/hPJXBuB.jpg" width="250">

# In[ ]:

uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')
del uxp_ratio
uxp.head()


# #### 2.3.3.5 Merge the final feature with uxp DataFrame
# The new feature will be merged with the uxp DataFrame (section 2.3.1) which keep all the features based on combinations of user-products. We perform a left join as we want to keep all the user-products that we have created on the uxp DataFrame
# 
# <img src="https://i.imgur.com/ObfHDPl.jpg" width="400">

# In[ ]:

uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')

#Fill NaN values
uxp = uxp.fillna(0)
uxp.head()

#Ratio of last 5 orders of each customer for a product
uxp ['last5_ ratio']= uxp.times_last5 / 5
uxp.head()

#Max days of 5 last orders for each product of users
uxp['max_days_last5'] = op5.groupby('user_id')[["days_since_prior_order"]].max()
uxp.head()

#Mean days of 5 last orders for each product of users
uxp['med_days_last5'] = op5.groupby('user_id')[["days_since_prior_order"]].median()
uxp.head()

#Max days days of orders for each product of users
uxp['uxp_max_days'] = op.groupby('user_id')[["days_since_prior_order"]].max()
uxp.head()

# #### 2.3.3.6 Fill NaN values
# If you check uxp DataFrame you will notice that some rows have NaN values for our new feature. This happens as there might be products that the customer did not buy on its last five orders. For these cases, we turn NaN values into zero (0) with .fillna(0) method.

# In[ ]:


uxp = uxp.fillna(0)
uxp.head()


# ## 2.4 Merge all features
# We now merge the DataFrames with the three types of predictors that we have created (i.e., for the users, the products and the combinations of users and products).
# 
# We will start from the **uxp** DataFrame and we will add the user and prd DataFrames. We do so because we want our final DataFrame (which will be called **data**) to have the following structure: 
# 
# <img style="float: left;" src="https://i.imgur.com/mI5BbFE.jpg" >
# 
# 
# 
# 
# 

# ### 2.4.1 Merge uxp with user DataFrame
# Here we select to perform a left join of uxp with user DataFrame based on matching key "user_id"
# 
# <img src="https://i.imgur.com/WlI84Ud.jpg" width="400">
# 
# Left join, ensures that the new DataFrame will have:
# - all the observations of the uxp (combination of user and products) DataFrame 
# - all the **matching** observations of user DataFrame with uxp based on matching key **"user_id"**
# 
# The new DataFrame as we have already mentioned, will be called **data**.

# In[ ]:


#Merge uxp features with the user features
#Store the results on a new DataFrame
data = uxp.merge(user, on='user_id', how='left')
data.head()


# ### 2.4.1 Merge data with prd DataFrame
# In this step we continue with our new DataFrame **data** and we perform a left join with prd DataFrame. The matching key here is the "product_id".
# <img src="https://i.imgur.com/Iak6nIz.jpg" width="400">
# 
# Left join, ensures that the new DataFrame will have:
# - all the observations of the data (features of userXproducts and users) DataFrame 
# - all the **matching** observations of prd DataFrame with data based on matching key **"product_id"**

# In[ ]:


#Merge uxp & user features (the new DataFrame) with prd features
data = data.merge(prd, on='product_id', how='left')
data.head()


# ### 2.4.2 Delete previous DataFrames

# The information from the DataFrames that we have created to store our features (op, user, prd, uxp) is now stored on **data**. 
# 
# As we won't use them anymore, we now delete them.

# In[ ]:


del op, user, prd, uxp, last_five, op5, total_orders
gc.collect()


# # 3. Create train and test DataFrames
# ## 3.1 Include information about the last order of each user
# 
# The **data** DataFrame that we have created on the previous chapter (2.4) should include two more columns which define the type of user (train or test) and the order_id of the future order.
# This information can be found on the initial orders DataFrame which was provided by Instacart: 
# 
# <img style="float: left;" src="https://i.imgur.com/jbatzRY.jpg" >
# 
# 
# Towards this end:
# 1. We select the **orders** DataFrame to keep only the future orders (labeled as "train" & "test). 
# 2. Keep only the columns of our desire ['eval_set', 'order_id'] <span style="color:red">**AND** </span> 'user_id' as is the matching key with our **data** DataFrame
# 2. Merge **data** DataFrame with the information for the future order of each customer using as matching key the 'user_id'

# To filter and select the columns of our desire on orders (the 2 first steps) there are numerous approaches:

# In[ ]:


## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)

## Second approach (if you want to test it you have to re-run the notebook):
# In one step keep only the future orders from all customers: train & test 
#orders_future = orders.loc[((orders.eval_set=='train') | (orders.eval_set=='test')), ['user_id', 'eval_set', 'order_id'] ]
#orders_future.head(10)

## Third approach (if you want to test it you have to re-run the notebook):
# In one step exclude all the prior orders so to deal with the future orders from all customers
#orders_future = orders.loc[orders.eval_set!='prior', ['user_id', 'eval_set', 'order_id'] ]
#orders_future.head(10)


# To fulfill step 3, we merge on **data** DataFrame the information for the last order of each customer. The matching key here is the user_id and we select a left join as we want to keep all the observations from **data** DataFrame.
# 
# <img src="https://i.imgur.com/m3pNVDW.jpg" width="400">

# In[ ]:


# bring the info of the future orders to data DF
data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# ## 3.2 Prepare the train DataFrame
# In order to prepare the train Dataset, which will be used to create our prediction model, we need to include also the response (Y) and thus have the following structure:
# 
# <img style="float: left;" src="https://i.imgur.com/PDu2vfR.jpg" >
# 
# Towards this end:
# 1. We keep only the customers who are labelled as "train" from the competition
# 2. For these customers we get from order_products_train the products that they have bought, in order to create the response variable (reordered:1 or 0)
# 3. We make all the required manipulations on that dataset and we remove the columns that are not predictors
# 
# So now we filter the **data** DataFrame so to keep only the train users:

# In[ ]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train']
data_train.head()


# For these customers we get from order_products_train the products that they have bought. The matching keys are here two: the "product_id" & "order_id". A left join keeps all the observations from data_train DataFrame
# 
# <img src="https://i.imgur.com/kndys9d.jpg" width="400">

# In[ ]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)


# On the last columm (reordered) you can find out our response (y). 
# There are combinations of User X Product which they were reordered (1) on last order where other were not (NaN value).
# 
# Now we manipulate the data_train DataFrame, to bring it into a structure for Machine Learning (X1,X2,....,Xn, y):
# - Fill NaN values with value zero (regards reordered rows without value = 1)

# In[ ]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# - Set as index the column(s) that describe uniquely each row (in our case "user_id" & "product_id")
# 

# In[ ]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# - Remove columns which are not predictors (in our case: 'eval_set','order_id')

# In[ ]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)


# ## 3.3 Prepare the test DataFrame
# The test DataFrame must have the same structure as the train DataFrame, excluding the "reordered" column (as it is the label that we want to predict).
# <img style="float: left;" src="https://i.imgur.com/lLJ7wpA.jpg" >
# 
#  To create it, we:
# - Keep only the customers who are labelled as test

# In[ ]:


#Keep only the future orders from customers who are labelled as test
data_test = data[data.eval_set=='test']
data_test.head()


# - Set as index the column(s) that uniquely describe each row (in our case "user_id" & "product_id")
# 

# In[ ]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id'])
data_test.head()


# - Remove the columns that are predictors (in our case:'eval_set', 'order_id')

# In[ ]:


#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()



# In[ ]:


# TRAIN FULL 
###########################
## IMPORT REQUIRED PACKAGES
###########################
import xgboost as xgb
from sklearn.model_selection import train_test_split
##########################################
## SPLIT DF TO: X_train, y_train (axis=1)
##########################################
#X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered
X_train, X_val, y_train, y_val = train_test_split(data_train.drop('reordered', axis=1), data_train.reordered, test_size=0.8, random_state=42)

########################################
## SET BOOSTER'S PARAMETERS
########################################
parameters = {'eval_metric':'logloss', 
              'max_depth': 5, 
              'colsample_bytree': 0.4,
              'subsample': 0.75,
             }

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)

########################################
## TRAIN MODEL
########################################
model = xgbc.fit(X_train, y_train)

##################################
# FEATURE IMPORTANCE - GRAPHICAL
##################################
#xgb.plot_importance(model)


# ## 4.2 Fine-tune your model
# 
# Most algorithms have their own parameters that we need to declare. With method .get_params() we can retrieve the parameters of our fitting model

# In[ ]:


model.get_xgb_params()


# These parameters do not necessarily create the best fitting model (in terms of prediction score). The method .GridSearchCV( ) can make several trials to define the best parameters for our fitting model. 

# In[ ]:

'''
###########################
## DISABLE WARNINGS
###########################
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

###########################
## IMPORT REQUIRED PACKAGES
###########################
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

####################################
## SET BOOSTER'S RANGE OF PARAMETERS
# IMPORTANT NOTICE: Fine-tuning an XGBoost model may be a computational prohibitive process with a regular computer or a Kaggle kernel. 
# Be cautious what parameters you enter in paramiGrid section.
# More paremeters means that GridSearch will create and evaluate more models.
####################################    
paramGrid = {"max_depth":[5,10,15],
           "colsample_bytree":[0.3,0.4,0.5]}  

########################################
## INSTANTIATE XGBClassifier()
########################################
xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', num_boost_round=10, gpu_id=0, tree_method = 'gpu_hist')

##############################################
## DEFINE HOW TO TRAIN THE DIFFERENT MODELS
#############################################
gridsearch = GridSearchCV(xgbc, paramGrid, cv=3, verbose=2, n_jobs=1)

################################################################
## TRAIN THE MODELS
### - with the combinations of different parameters
### - here is where GridSearch will be exeucuted
#################################################################
model = gridsearch.fit(X_train, y_train)

##################################
## OUTPUT(S)
##################################
# Print the best parameters
print("The best parameters are: /n",  gridsearch.best_params_)

# Store the model for prediction (chapter 5)
model = gridsearch.best_estimator_

# Delete X_train , y_train
del [X_train, y_train]


# The model has now the new parameters from GridSearchCV:

# In[ ]:


model.get_params()
'''

# # 5. Apply predictive model (predict)
# The model that we have created is stored in the **model** object.
# At this step we predict the values for the test data and we store them in a new column in the same DataFrame.
# 
# For better results, we set a custom threshold to 0.21. The best custom threshold can be found through a grid search.

# In[ ]:


'''
# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
test_pred = model.predict(data_test).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array
'''


# In[ ]:


## OR set a custom threshold (in this problem, 0.21 yields the best prediction)
test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array


# In[ ]:


#Save the prediction (saved in a numpy array) on a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head(10)


# In[ ]:


# Reset the index
final = data_test.reset_index()
# Keep only the required columns to create our submission file (for chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()


# # 6. Creation of Submission File
# To submit our prediction to Instacart competition we have to get for each user_id (test users) their last order_id. The final submission file should have the test order numbers and the products that we predict that are going to be bought.
# 
# To create this file we retrieve from orders DataFrame all the test orders with their matching user_id:

# In[ ]:


orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]
orders_test.head()


# We merge it with our predictions (from chapter 5) using a left join:
# <img src="https://i.imgur.com/KJubu0v.jpg" width="400">

# In[ ]:


final = final.merge(orders_test, on='user_id', how='left')
final.head()


# And we move on with two final manipulations:
# - remove any undesired column (in our case user_id)

# In[ ]:


#remove user_id column
final = final.drop('user_id', axis=1)


# - set product_id column as integer (mandatory action to proceed to the next step)

# In[ ]:


#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

## Remove all unnecessary objects
del orders
del orders_test
gc.collect()

final.head()


# For our submission file we initiate an empty dictionary. In this dictionary we will place as index the order_id and as values all the products that the order will have. If none product will be purchased, we have explicitly to place the string "None". This syntax follows the submission's file standards defined by the competition.

# In[ ]:


d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in final.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()

#We now check how the dictionary were populated (open hidden output)
d


# We convert the dictionary to a DataFrame and prepare it to extact it into a .csv file

# In[ ]:


#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()


# **The submission file should have 75.000 predictions to be submitted in the competition**

# In[ ]:


#Check if sub file has 75000 predictions
sub.shape[0]
print(sub.shape[0]==75000)


# The DataFrame can now be converted to .csv file. Pandas can export a DataFrame to a .csv file with the .to_csv( ) function.

# In[7]:


sub.to_csv('sub.csv', index=False)


