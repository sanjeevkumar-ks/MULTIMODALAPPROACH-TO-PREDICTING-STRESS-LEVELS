
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline     
sns.set(color_codes=True)

df = pd.read_csv("mexican_medical_students_mental_health_data.csv")
# To display the top 5 rows 
print('-------dataset values ---------------------------')
print(df.head(5))    
print('-------datatypes of all columns---------------------------')
print(df.dtypes)
print('-------duplicate values count---------------------------')
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
print('-------values count---------------------------')
print(df.count())

print('-------missing or null values count---------------------------')
print(df.isnull().sum())
print('-------dropping null values ---------------------------')
df = df.dropna()    # Dropping the missing values.
print(df.count())
print('-------after dropping null values- null values count -----------------')
print(df.isnull().sum())   # After dropping the values

print('-------Detecting Outliers---------------------------------------------')
sns.boxplot(x=df['previous_depression_treatment'])

plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
print(c)      

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['times_week_nap'], df['grades'])
ax.set_xlabel('times_week_nap')
ax.set_ylabel('grades')
plt.show()