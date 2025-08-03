import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load your frozen splits
train = pd.read_csv('sms_train.csv')
val = pd.read_csv('sms_val.csv')
test = pd.read_csv('sms_test.csv')

# 2. Create numerical labels (0 for 'ham', 1 for 'spam') if not present
label_map = {'ham': 0, 'spam': 1}
for df in [train, val, test]:
    df['label'] = df['Category'].map(label_map)

# 3. Preprocessing: Lowercase and strip whitespace
def preprocess(text):
    return str(text).lower().strip()

for df in [train, val, test]:
    df['Message'] = df['Message'].apply(preprocess)

# -- LOGISTIC REGRESSION BASELINE --
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['Message'])
X_val   = vectorizer.transform(val['Message'])
X_test  = vectorizer.transform(test['Message'])

y_train = train['label']
y_val   = val['label']
y_test  = test['label']

clf = LogisticRegression(max_iter=1000, class_weight='balanced')


clf.fit(X_train, y_train)

val_preds = clf.predict(X_val)
print("\nValidation set performance (LogReg):")
print(classification_report(y_val, val_preds, target_names=['ham', 'spam']))

test_preds = clf.predict(X_test)
print("\nTest set performance (LogReg):")
print(classification_report(y_test, test_preds, target_names=['ham', 'spam']))

print("Confusion Matrix (LogReg, test set):\n", confusion_matrix(y_test, test_preds))


# -- ROBERTA TRANSFORMER CLASSIFIER --

from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

import torch
from torch.utils.data import Dataset

class SMSDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['Message'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]   # This is now always an int, not a string
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = SMSDataset(train, tokenizer)
val_dataset = SMSDataset(val, tokenizer)
test_dataset = SMSDataset(test, tokenizer)

from transformers import Trainer, TrainingArguments

print('Transformers version:', __import__('transformers').__version__)

training_args = TrainingArguments(
    output_dir='./roberta_sms',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
    report_to='none'
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

import torch
from torch.nn import CrossEntropyLoss

# Compute weights: inverse of class frequencies
num_spam = sum(train['label'] == 1)
num_ham = sum(train['label'] == 0)
weights = torch.tensor([1.0 / num_ham, 1.0 / num_spam], dtype=torch.float)
weights = weights / weights.sum()  # normalize

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

def compute_loss(self, model, inputs, return_outputs=False, **kwargs):   # add **kwargs
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")

    if self.class_weights is not None:
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
    else:
        loss = outputs.loss

    return (loss, outputs) if return_outputs else loss

# Compute class weights:
num_spam = sum(train['label'] == 1)
num_ham = sum(train['label'] == 0)
weights = torch.tensor([1.0 / num_ham, 1.0 / num_spam], dtype=torch.float)
weights = weights / weights.sum()

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=weights,
)

trainer.train()


trainer.train()

# Evaluate on the test set
test_output = trainer.predict(test_dataset)
print("\nTest Metrics (RoBERTa):", test_output.metrics)

import numpy as np
test_preds = np.argmax(test_output.predictions, axis=1)
print("Classification Report (RoBERTa):\n", classification_report(test['label'], test_preds, target_names=['ham', 'spam']))
print("Confusion Matrix (RoBERTa, test set):\n", confusion_matrix(test['label'], test_preds))

# Confirm your data columns for debug purposes
print("Train columns:", train.columns)
print("Val columns:", val.columns)
print("Test columns:", test.columns)
