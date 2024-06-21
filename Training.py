import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Load the dataset
df = pd.read_csv('collegePlace.csv')

# Encode categorical variables
label_encoder_gender = LabelEncoder()
label_encoder_gender.fit(df['Gender'])

label_encoder_stream = LabelEncoder()
label_encoder_stream.fit(df['Stream'])

df['Gender'] = label_encoder_gender.transform(df['Gender'])
df['Stream'] = label_encoder_stream.transform(df['Stream'])

# Define features and target variable
X = df.drop('PlacedOrNot', axis=1)
y = df['PlacedOrNot']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10, None],
    'criterion': ['gini', 'entropy']
}

# Initialize RandomForest
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Display the best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# Initialize the model with best parameters
best_rf = grid_search.best_estimator_

# Train the model
best_rf.fit(X_train, y_train)

# Make predictions
y_pred = best_rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'RandomForest Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(best_rf, X, y, cv=skf, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

# Initialize other models with updated parameters
svc = SVC(probability=True, random_state=42)
gbc = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

# Train and evaluate SVM
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)
print(f'SVM Accuracy: {svc_accuracy:.2f}')

# Train and evaluate Gradient Boosting
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
gbc_accuracy = accuracy_score(y_test, gbc_pred)
print(f'Gradient Boosting Accuracy: {gbc_accuracy:.2f}')

# Train and evaluate XGBoost
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'XGBoost Accuracy: {xgb_accuracy:.2f}')

# Initialize VotingClassifier with best models
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_rf),
    ('svc', svc),
    ('gbc', gbc),
    ('xgb', xgb)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions with the ensemble model
ensemble_pred = ensemble_model.predict(X_test)

# Calculate accuracy
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f'Ensemble Model Accuracy: {ensemble_accuracy:.2f}')


# After training the ensemble_model
ensemble_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder_gender, 'label_encoder_gender.pkl')
joblib.dump(label_encoder_stream, 'label_encoder_stream.pkl')



