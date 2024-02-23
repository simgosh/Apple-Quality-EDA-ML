#Importing all relevance libs 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Importing dataset with csv & Explore & Manipulate 
apple = pd.read_csv("apple_quality.csv")
print(apple.info())
print(apple.head())

#Convert from object to float "Acidity" column : 
apple["Acidity"] = pd.to_numeric(apple["Acidity"], errors="coerce"  )
print(apple.dtypes)

apple["Quality"] = apple["Quality"].astype("category")
print(apple["Quality"])
# Missing data 
print(apple.isnull().sum().sort_values())
print(apple.dropna(inplace=True))
print(apple.isnull().sum())

print(apple.duplicated().sum())
print(apple.shape)

#Dropping A_id (same as indexes)
apple.drop("A_id", axis=1, inplace=True)
print(apple.head())

#Visualization step 1: 
plt.figure(figsize=(10,4))
palette = sns.color_palette("bright")
sns.countplot(x=apple["Quality"], data=apple, palette=palette)
plt.tight_layout()
plt.ylabel("Count")
plt.title("Apple Quality")
plt.show()

#Apple quality according to "taste" 
taste = ["Weight","Sweetness", "Crunchiness", "Juiciness","Ripeness", "Acidity"]
for i in taste: 
    plt.scatter(apple[i], apple["Size"])
    plt.set_cmap('coolwarm')
    plt.xlabel(i)
    plt.ylabel("Size")
    plt.title(f"Scatter plot : {i} vs. Size")
    plt.show()
    
# from matplotlib import colormaps
# print(list(colormaps))

#Size vs Weight reg plot : 
plt.figure(figsize=(12,4))
sns.regplot(x="Weight", y="Size", data=apple,scatter_kws={"color":"black", "alpha":0.4} ,line_kws= {"color": "red"})
plt.xlabel("Weight")
plt.ylabel("Size")
plt.title("Size vs. Weight")
plt.show()


#Sweetness vs Weight plot 
plt.figure(figsize=(4,4))
sns.scatterplot(x="Sweetness", y="Weight", data=apple, alpha=0.5)
plt.show()


##Correlation between features using heatmap (1st "Quality" to convert from obj to int(1,0))
apple["Quality"] = apple["Quality"].str.replace('good', "1").str.replace("bad", "0").astype(int)

#Correlation between features using heatmap
plt.figure(figsize=(4,4))
sns.heatmap(apple.corr(), annot=True, cmap="Greens")
plt.show()



#For now this we can use different models:
# Logistic Regression
# Decision Tree
# Ensemble algorithms: Random Forest


#LogisticRegression 
X=apple.drop("Quality", axis=1)
y = apple["Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_log= logreg.predict(X_test)

print(accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print(accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
