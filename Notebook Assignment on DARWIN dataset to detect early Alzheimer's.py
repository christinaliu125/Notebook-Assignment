#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from datascience import *
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns


# # Interpreting Data

# In[16]:


darwin_table = pd.read_csv("DARWIN.csv")
darwin_table


# In[17]:


type(darwin_table)
darwin_table.head()


# In[18]:


#summary statistics
darwin_table.describe()


# In[19]:


darwin_table.info()


# In[23]:


darwin_table["class"]

#replace P with patient and H with healthy
darwin_table.replace(to_replace={'P': 'Patient', 'H': 'Healthy'}, value =None, inplace=True)

darwin_table['class']


# In[20]:


#find the number of healthy people in the dataset
darwin_table['class'].value_counts()

number_healthy = 85
#find the number of patients(with Alzheimer's)
number_w_alz = 89


# # Visual Representations of Data

# In[61]:


#visual representations of the data


# In[67]:


#Create a scatter plot that compares the mean speed in air and mean speed on paper for
#patients and healthy participants


# In[95]:


'''!pip check pandas

x = darwin_table['mean_speed_in_air25'].where('class' == 'Healthy')
y = np.array(darwin_table['mean_speed_on_paper25'].where('class' == 'Healthy'))
             
plt.scatter(x, y, color = 'hotpink')'''


# # Find Most Significant Features

# In[4]:


from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Correlation is a measure of the linear relationship of 2 or more variables. Through correlation, we can predict one variable from the other. The logic behind using correlation for feature selection is that the good variables are highly correlated with the target. Furthermore, variables should be correlated with the target but should be uncorrelated among themselves.
# 
# If two variables are correlated, we can predict one from the other. Therefore, if two features are correlated, the model only really needs one of them, as the second one does not add additional information. We will use the Pearson Correlation here.

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


#correlation matrix

#load columns 1-18
darwin_table_columns = darwin_table.columns

print(darwin_table_columns[1:19])

cor_task1 = darwin_table.iloc[0:5, 1:19].corr()

print(cor_task1)


# In[70]:


print(cor_task1 > abs(0.9))


# In[65]:


#plotting heatmap
plt.figure(figsize = (100,20))
sns.heatmap(cor, annot = True)


# In[92]:


#information gain
X_task1 = darwin_table.iloc[0:, 1:19]
Y_task1 = darwin_table.iloc[0:, -1]

print(X_task1)
print(Y_task1)
importances = mutual_info_classif(X_task1, Y_task1)
feat_importances = pd.Series(importances, darwin_table.columns[1:19])
print(feat_importances)
feat_importances.plot(kind='barh', color='teal')
plt.show()


# In[94]:


#information gain
X_task2 = darwin_table.iloc[0:, 19:37]
Y_task2 = darwin_table.iloc[0:, -1]

print(X_task1)
print(Y_task1)
importances = mutual_info_classif(X_task2, Y_task2)
feat_importances = pd.Series(importances, darwin_table.columns[1:19])
print(feat_importances)
feat_importances.plot(kind='barh', color='teal')
plt.show()


# In[ ]:





# # Try to load data with numpy

# In[24]:


np_darwin_table = Table().read_table("DARWIN.csv")
np_darwin_table 


# # Data Visualization

# In[119]:


x = np_darwin_table.where('class', are.equal_to('H')).column('mean_speed_in_air25')
y = np_darwin_table.where('class', are.equal_to('H')).column('mean_speed_on_paper25')

plt.scatter(x, y, color = 'hotpink')

#print(x, y)

x = np_darwin_table.where('class', are.equal_to('P')).column('mean_speed_in_air25')
y = np_darwin_table.where('class', are.equal_to('P')).column('mean_speed_on_paper25')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[71]:


#air time and total time
x = np_darwin_table.where('class', are.equal_to('H')).column('air_time1')
y = np_darwin_table.where('class', are.equal_to('H')).column('total_time1')

plt.scatter(x, y, color = 'hotpink')


x = np_darwin_table.where('class', are.equal_to('P')).column('air_time1')
y = np_darwin_table.where('class', are.equal_to('P')).column('total_time1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[83]:


#using information gain
x = np_darwin_table.where('class', are.equal_to('H')).column('pressure_mean1')
y = np_darwin_table.where('class', are.equal_to('H')).column('gmrt_on_paper1')

plt.scatter(x, y, color = 'hotpink')


x = np_darwin_table.where('class', are.equal_to('P')).column('pressure_mean1')
y = np_darwin_table.where('class', are.equal_to('P')).column('gmrt_on_paper1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[122]:


x = np_darwin_table.where('class', are.equal_to('H')).column('max_x_extension1')
y = np_darwin_table.where('class', are.equal_to('H')).column('max_y_extension1')

plt.scatter(x, y, color = 'hotpink')


x = np_darwin_table.where('class', are.equal_to('P')).column('max_x_extension1')
y = np_darwin_table.where('class', are.equal_to('P')).column('max_y_extension1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[123]:


x = np_darwin_table.where('class', are.equal_to('H')).column('num_of_pendown1')
y = np_darwin_table.where('class', are.equal_to('H')).column('pressure_mean1')

plt.scatter(x, y, color = 'hotpink')


x = np_darwin_table.where('class', are.equal_to('P')).column('num_of_pendown1')
y = np_darwin_table.where('class', are.equal_to('P')).column('pressure_mean1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[125]:


x = np_darwin_table.where('class', are.equal_to('H')).column('mean_jerk_in_air1')
y = np_darwin_table.where('class', are.equal_to('H')).column('mean_jerk_on_paper1')

plt.scatter(x, y, color = 'hotpink')

#print(x, y)

x = np_darwin_table.where('class', are.equal_to('P')).column('mean_jerk_in_air1')
y = np_darwin_table.where('class', are.equal_to('P')).column('mean_jerk_on_paper1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[126]:


x = np_darwin_table.where('class', are.equal_to('H')).column('pressure_mean1')
y = np_darwin_table.where('class', are.equal_to('H')).column('pressure_var1')

plt.scatter(x, y, color = 'hotpink')

#print(x, y)

x = np_darwin_table.where('class', are.equal_to('P')).column('pressure_mean1')
y = np_darwin_table.where('class', are.equal_to('P')).column('pressure_var1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[25]:


x = np_darwin_table.where('class', are.equal_to('H')).column('gmrt_in_air1')
y = np_darwin_table.where('class', are.equal_to('H')).column('gmrt_on_paper1')

plt.scatter(x, y, color = 'hotpink')

#print(x, y)

x = np_darwin_table.where('class', are.equal_to('P')).column('gmrt_in_air1')
y = np_darwin_table.where('class', are.equal_to('P')).column('gmrt_on_paper1')

plt.scatter(x, y, color = '#88c999')

plt.show()


# In[41]:


darwin_table.columns


# In[88]:


#diff types of visualizations


# In[91]:


#boxplot

plt.boxplot(X_task1)


# In[ ]:





# # Data Cleaning

# In[66]:


#look for null values
print(darwin_table.isnull().sum() == 0)


# # Analysis

# What patterns or trends can be seen in the data?
# 

# Not many clear patterns and trends can be seen from the above visualizations. However, for each task, different features have been shown to give different amounts of information. 
# 
# 
# 

# In[ ]:




