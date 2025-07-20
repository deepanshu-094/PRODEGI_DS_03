import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('bank-full.csv', sep=';')

# Show basic info
print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Class distribution:")
print(df['y'].value_counts())

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for column in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le  # Save for inverse_transform if needed

# Features and target
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"🔹 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=label_encoders['y'].classes_, filled=True)
plt.title("Decision Tree for Term Deposit Prediction")
plt.show()
