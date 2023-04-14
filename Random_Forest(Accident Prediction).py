#Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv(r"F:\Random Forest\Data.csv")
#print(data)

#replacing strings with specific numeric values
data['Day_Night'] = data['Day_Night'].replace({'Day': 0, 'Night': 1})
data['Crossing'] = data['Crossing'].replace({'TRUE': 0, 'FALSE': 1})
data['Junction'] = data['Junction'].replace({'TRUE': 0, 'FALSE': 1})
data['Railway'] = data['Railway'].replace({'TRUE': 0, 'FALSE': 1})
data['Traffic_Signal'] = data['Traffic_Signal'].replace({'TRUE': 0, 'FALSE': 1})
data['Turning_Loop'] = data['Turning_Loop'].replace({'TRUE': 0, 'FALSE': 1})
data['Bump'] = data['Bump'].replace({'TRUE': 0, 'FALSE': 1})

# Split the dataset into features and target variable
X = data[['Temperature', 'Wind_Speed', 'Visibility', 'Bump', 'Crossing', 'Junction', 'Railway', 'Traffic_Signal', 'Turning_Loop', 'Precipitation', 'Day_Night']]  # Features
y = data['Severity']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#print(data)

#imputing missing datas in any of the incomplete columns
impute1 = SimpleImputer(strategy='mean')
impute2 = SimpleImputer(strategy='median')
impute3 = SimpleImputer(strategy='most_frequent')


X_train_TempImp = impute3.fit_transform(X_train[['Temperature']])
X_test_TempImp = impute3.transform(X_test[['Temperature']])
X_train[['Temperature']] = X_train_TempImp
X_test[['Temperature']] = X_test_TempImp

X_train_WindImp = impute1.fit_transform(X_train[['Wind_Speed']])
X_test_WindImp = impute1.transform(X_test[['Wind_Speed']])
X_train[['Wind_Speed']] = X_train_WindImp
X_test[['Wind_Speed']] = X_test_WindImp

X_train_VisImp = impute2.fit_transform(X_train[['Visibility']])
X_test_VisImp = impute2.transform(X_test[['Visibility']])
X_train[['Visibility']] = X_train_VisImp
X_test[['Visibility']] = X_test_VisImp

X_train_PrepImp = impute3.fit_transform(X_train[['Precipitation']])
X_test_PrepImp = impute3.transform(X_test[['Precipitation']])
X_train[['Precipitation']] = X_train_PrepImp
X_test[['Precipitation']] = X_test_PrepImp


X_train_DnImp = impute3.fit_transform(X_train[['Day_Night']])
X_test_DnImp = impute3.transform(X_test[['Day_Night']])
X_train[['Day_Night']] = X_train_DnImp
X_test[['Day_Night']] = X_test_DnImp

# Initializing the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

print('Model Training Started. PLEASE WAIT....')
# Train the model
rf_model.fit(X_train, y_train)

print('Making test predictions')
# Making predictions on the testing set
y_pred = rf_model.predict(X_test)

# Evaluating the model
acc = accuracy_score(y_test, y_pred)
acc = acc*100
print('The model ran with an accuracy of ', acc, '%')

print('The model is ready to predict accidents')

new_input = pd.DataFrame({
    'Temperature': [42.1],
    'Wind_Speed': [1],
    'Visibility': [5],
    'Bump': [0],
    'Crossing': [1],
    'Junction': [1],
    'Railway': [0],
    'Traffic_Signal': [1],
    'Turning_Loop': [0],
    'Precipitation': [0.05],
    'Day_Night': [0]
     })

print('The new given input')
print(new_input)
predictions = rf_model.predict(new_input)
print('Accident Severity is(US Standard):', predictions)