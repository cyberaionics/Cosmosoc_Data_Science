import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')    

print(train.isnull().sum())

test_passenger_ids = test['PassengerId'].copy()

numeric_cols = train.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Transported':
        train[col].fillna(train[col].mode()[0], inplace=True)
        test[col].fillna(test[col].mode()[0], inplace=True)

def safe_label_encode(train_col, test_col):
    le = LabelEncoder()
    
    combined = pd.concat([train_col, test_col]).astype(str)
    le.fit(combined)
    
    train_encoded = le.transform(train_col.astype(str))
    test_encoded = le.transform(test_col.astype(str))
    
    return train_encoded, test_encoded

for col in categorical_cols:
    if col != 'Transported':
        train[col], test[col] = safe_label_encode(train[col], test[col])



X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)  



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")


predictions = model.predict(test)

submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Transported': predictions.astype(bool)
})
submission.to_csv('Gradient Descenders Predicted.csv', index=False)
