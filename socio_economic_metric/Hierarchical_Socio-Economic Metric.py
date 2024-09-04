#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import AgglomerativeClustering


# In[2]:


df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\Country-data.csv")
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df['country'].nunique()


# In[8]:


df1=df


# In[9]:


df


# ### Exploratory Data Analytics

# In[10]:


plt.figure(figsize = (30,5))
child_mort = df[['country','child_mort']].sort_values('child_mort', ascending = False)
ax = sns.barplot(x='country', y='child_mort', data= child_mort)
ax.set(xlabel = '', ylabel= 'Child Mortality Rate')
plt.xticks(rotation=90)
plt.show()


# We are able to see how Child Mortality Rate is distributed across the all countries. Focus on the objective of the task

# In[11]:


plt.figure(figsize = (10,5))
child_mort_top10 = df[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)
ax = sns.barplot(x='country', y='child_mort', data= child_mort_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Child Mortality Rate')
plt.xticks(rotation=90)
plt.show()


# Top 10 Countries having highest Child Mortality Rate are present in Africa having poor healthcare facilities.

# In[12]:


plt.figure(figsize = (30,5))
total_fer = df[['country','total_fer']].sort_values('total_fer', ascending = False)
ax = sns.barplot(x='country', y='total_fer', data= total_fer)
ax.set(xlabel = '', ylabel= 'Fertility Rate')
plt.xticks(rotation=90)
plt.show()


# We are able to see how Fertility Rate is distributed across the all countries.

# In[13]:


plt.figure(figsize = (10,5))
total_fer_top10 = df[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)
ax = sns.barplot(x='country', y='total_fer', data= total_fer_top10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Fertility Rate')
plt.xticks(rotation=90)
plt.show()


# Top 10 Countries having highest Fertility Rate are places where people are poorest in all.

# In[14]:


plt.figure(figsize = (32,5))
life_expec = df[['country','life_expec']].sort_values('life_expec', ascending = True)
ax = sns.barplot(x='country', y='life_expec', data= life_expec)
ax.set(xlabel = '', ylabel= 'Life Expectancy')
plt.xticks(rotation=90)
plt.show()


# We are able to see how Life Expectancy is distributed across the all countries. Focus on the objective of the task.

# In[15]:


plt.figure(figsize = (10,5))
life_expec_bottom10 = df[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)
ax = sns.barplot(x='country', y='life_expec', data= life_expec_bottom10)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Life Expectancy')
plt.xticks(rotation=90)
plt.show()


# Top 10 Countries having lowest Life Expectancy are places where healthcare system is not available or efficient.

# In[16]:


plt.figure(figsize = (32,5))
health = df[['country','health']].sort_values('health', ascending = True)
ax = sns.barplot(x='country', y='health', data= health)
ax.set(xlabel = '', ylabel= 'Health')
plt.xticks(rotation=90)
plt.show()


# We are able to see how Total health spending is distributed across the all countries. Focus on the objective of the task.

# In[17]:


fig, axs = plt.subplots(3,3,figsize = (18,18))

# Child Mortality Rate : Death of children under 5 years of age per 1000 live births

top5_child_mort = df[['country','child_mort']].sort_values('child_mort', ascending = False).head()
ax = sns.barplot(x='country', y='child_mort', data= top5_child_mort, ax = axs[0,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Child Mortality Rate')


top5_total_fer = df[['country','total_fer']].sort_values('total_fer', ascending = False).head()
ax = sns.barplot(x='country', y='total_fer', data= top5_total_fer, ax = axs[0,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Fertility Rate')



bottom5_life_expec = df[['country','life_expec']].sort_values('life_expec', ascending = True).head()
ax = sns.barplot(x='country', y='life_expec', data= bottom5_life_expec, ax = axs[0,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Life Expectancy')


bottom5_health = df[['country','health']].sort_values('health', ascending = True).head()
ax = sns.barplot(x='country', y='health', data= bottom5_health, ax = axs[1,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Health')


bottom5_gdpp = df[['country','gdpp']].sort_values('gdpp', ascending = True).head()
ax = sns.barplot(x='country', y='gdpp', data= bottom5_gdpp, ax = axs[1,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'GDP per capita')


bottom5_income = df[['country','income']].sort_values('income', ascending = True).head()
ax = sns.barplot(x='country', y='income', data= bottom5_income, ax = axs[1,2])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Per capita Income')


top5_inflation = df[['country','inflation']].sort_values('inflation', ascending = False).head()
ax = sns.barplot(x='country', y='inflation', data= top5_inflation, ax = axs[2,0])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Inflation')


bottom5_exports = df[['country','exports']].sort_values('exports', ascending = True).head()
ax = sns.barplot(x='country', y='exports', data= bottom5_exports, ax = axs[2,1])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel = '', ylabel= 'Exports')


bottom5_imports = df[['country', 'imports']].sort_values('imports', ascending=True).head()
ax = sns.barplot(x='country', y='imports', data=bottom5_imports, ax=axs[2, 2])

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel='', ylabel='Imports')


for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.tight_layout()
plt.savefig('EDA')
plt.show()


# In[18]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(numeric_only=True), annot = True, cmap="rainbow")
plt.savefig('Correlation')
plt.show()


# - child_mortality and life_expentency are highly correlated with correlation of -0.89
# - child_mortality and total_fertility are highly correlated with correlation of 0.85
# - imports and exports are highly correlated with correlation of 0.99
# - life_expentency and total_fertility are highly correlated with correlation of -0.76

# In[19]:


sns.pairplot(df,corner=True,diag_kind="kde")
plt.show()


# ### Data Preparation

# In[20]:


# Converting exports,imports and health spending percentages to absolute values.

df['exports'] = df['exports'] * df['gdpp']/100
df['imports'] = df['imports'] * df['gdpp']/100
df['health'] = df['health'] * df['gdpp']/100


# In[21]:


df


# In[22]:


# Dropping Country field as final dataframe will only contain data columns

df_drop = df.copy()
country = df_drop.pop('country')


# In[23]:


df_drop.head()


# ### Rescaling the Features

# In[24]:


# Standarisation technique for scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_drop)


# In[25]:


df_scaled


# ## Model Building

# ### Agglomerative Hierarchical Clustering

# #### Single Linkage:

# In[26]:


# Single linkage

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(14,4))
mergings = linkage(df_scaled, method='single', metric='euclidean')
dendrogram(mergings)
plt.show()


# #### Complete Linkage

# In[27]:


# Complete Linkage

plt.figure(figsize=(14,8))
mergings = linkage(df_scaled, method='complete',metric='euclidean')
dendrogram(mergings)
plt.show()


# In[28]:


dff = pd.DataFrame(df_scaled, columns=df_drop.columns)


# In[29]:


hc = AgglomerativeClustering(n_clusters=5, metric='euclidean',linkage='complete')


# In[30]:


hc


# In[31]:


labels_hc = hc.fit_predict(dff)


# In[32]:


labels_hc


# In[47]:


plt.figure(figsize=(10, 8))

# Plot the first scatter plot
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.scatter(dff.iloc[:, 4], dff.iloc[:, 6], c=labels_hc, cmap='rainbow')
plt.title('Scatter plot of Feature 5 vs Feature 7')
plt.xlabel('Feature 5')
plt.ylabel('Feature 7')

unique_labels = list(set(labels_hc))
for label in unique_labels:
    plt.scatter([], [], color=plt.cm.rainbow(label / max(unique_labels)), label=f'Cluster {label}')

# Adding the legend to the plot
plt.legend(title="Clusters", loc='upper right')

# Plot the second scatter plot
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.scatter(dff.iloc[:, 1], dff.iloc[:, 4], c=labels_hc, cmap='rainbow')
plt.title('Scatter plot of Feature 2 vs Feature 5')
plt.xlabel('Feature 2')
plt.ylabel('Feature 5')

# Create the legend manually
unique_labels = list(set(labels_hc))
for label in unique_labels:
    plt.scatter([], [], color=plt.cm.rainbow(label / max(unique_labels)), label=f'Cluster {label}')

# Adding the legend to the plot
plt.legend(title="Clusters", loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()


# In[49]:


cluster_0 =df[labels_hc==0]
cluster_0.head()


# In[50]:


cluster_4 =df[labels_hc==4]
cluster_4.head()


# In[53]:


df_append = pd.concat([cluster_0, cluster_4])
df_append.head()


# In[54]:


df_append.describe()


# We have removed few countries during outlier treatment but we might have dropped some countries which might be in need of help. Let's iterate our final list based on the information from the clusters which were in need of aid.ie, Cluster 0 and Cluster 4

# ### Number of Clusters in Hierarchical Clustering is 5

# In[ ]:
import pickle


with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save cluster labels and centroids for nearest neighbor search
cluster_data = pd.DataFrame(df_scaled, columns=df_drop.columns)
cluster_data['cluster'] = labels_hc
cluster_centers = cluster_data.groupby('cluster').mean().values

with open('cluster_centers.pkl', 'wb') as f:
    pickle.dump(cluster_centers, f)

with open('cluster_labels.pkl', 'wb') as f:
    pickle.dump(labels_hc, f)
