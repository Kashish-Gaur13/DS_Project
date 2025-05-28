import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import preprocess

# Load preprocessed data
df = preprocess()

# Drop NA and encode categorical data
df.dropna(inplace=True)
# Inspect sample responses to define churn logic
print("Sample responses from survey:")
print(df['response'].dropna().astype(str).sample(20))

# Encode categorical features
le_kasse = LabelEncoder()
df['krankenkasse'] = le_kasse.fit_transform(df['krankenkasse'])
le_question = LabelEncoder()
df['question'] = le_question.fit_transform(df['question'])

# Simulate churn column for demo (real data needs true churn labels)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df['churn'] = df['response'].apply(lambda x: 1 if x < 1.0 else 0)
print("Churn distribution:")
print(df['churn'].value_counts())
print(df[['response', 'churn']].corr())
# Select features
features = ['krankenkasse', 'question', 'response', 'marktanteil_mitglieder', 'risikofaktor', 'avg_zusatzbeitrag']
df = df[features + ['churn']]
df['response'] = pd.to_numeric(df['response'], errors='coerce').fillna(0)

X = df.drop(columns=['churn', 'response'])
y = df['churn']
print(y.value_counts())
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print ("Churn label distribution:")
print(y.value_counts())
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.show()
