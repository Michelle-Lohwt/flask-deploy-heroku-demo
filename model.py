# Import Libraries
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load csv file
df = pd.read_csv('Iris.csv')

# Seperate independent and dependent variable
X = df.iloc[:, 1:-1] # eliminate 'Id' column
y = df['Species']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit model
classifier.fit(X_train, y_train)

# Make pickle file
pickle.dump(classifier, open("model.pkl", "wb"))