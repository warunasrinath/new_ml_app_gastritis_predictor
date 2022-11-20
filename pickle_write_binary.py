# First importing essential libraries 
from re import X
import numpy as np
import pandas as pd
import pickle

# Load the CSV dataset
df = pd.read_csv('Gastritis Dataset.csv')

#Renaming column names
df = df.rename(columns={'The_sign_of_Gastritis':'TSG'})
df = df.rename(columns={'Time_seen_for_signs':'TSS'})
df = df.rename(columns={'water_consumption_per_day':'WCP'})
df = df.rename(columns={'Practices_for_the_signs':'PFS'})
df = df.rename(columns={'Gastric_irritation_drugs':'GID'})
df = df.rename(columns={'Not_washing_hands_before_eating':'NWHB'})
df = df.rename(columns={'Skipped_and_delayed_meals':'SDM'})
df = df.rename(columns={'Eating_uncooked_foods':'EUF'})
df = df.rename(columns={'Involvement_in_Substance_Use':'ISU'})

#Model Building 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = df.drop(columns='Gastritis_symptoms')
y = df['Gastritis_symptoms']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#Then Creat Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

#Accuracy of training data
X_train_prediction = classifier.predict(X_train)
Accuracy_of_the_training_data =accuracy_score(X_train_prediction, y_train)
print('Accuracy of the training data: ', Accuracy_of_the_training_data)

# Creating a pickle file for the classifier
filename = 'gastritis_dataset.pkl'
pickle.dump(classifier, open(filename, 'wb'))