import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 1. Load datasets
train = pd.read_csv('sms_train.csv')
val = pd.read_csv('sms_val.csv')
test = pd.read_csv('sms_test.csv')

# 2. Create numerical labels (0 = ham, 1 = spam)
label_map = {'ham': 0, 'spam': 1}
for df in [train, val, test]:
    df['label'] = df['Category'].map(label_map)

# 3. Preprocessing: lowercase and strip whitespace
def preprocess(text):
    return str(text).lower().strip()

for df in [train, val, test]:
    df['Message'] = df['Message'].apply(preprocess)

# 4. Vectorization: TF-IDF (fit only on train)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['Message'])
X_val = vectorizer.transform(val['Message'])
X_test = vectorizer.transform(test['Message'])

y_train = train['label']
y_val = val['label']
y_test = test['label']

# 5. Hyperparameter tuning for SVM regularization parameter C using validation set
best_model = None
best_f1 = -1
C_values = [0.01, 0.1, 1, 10, 100]

print("Tuning SVM hyperparameter C:")
for C in C_values:
    clf = SVC(C=C, kernel='linear', class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    f1 = f1_score(y_val, val_preds)
    print(f"C={C} -> Validation F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model = clf

print(f"\nBest C based on validation F1: {best_model.C}")

# 6. Evaluate on validation set with best model
val_preds = best_model.predict(X_val)
print("\nValidation Set Performance (SVM):")
print(classification_report(y_val, val_preds, target_names=['ham', 'spam']))

# 7. Evaluate on test set
test_preds = best_model.predict(X_test)
print("\nTest Set Performance (SVM):")
print(classification_report(y_test, test_preds, target_names=['ham', 'spam']))
print("Confusion Matrix (SVM, test set):")
print(confusion_matrix(y_test, test_preds))
