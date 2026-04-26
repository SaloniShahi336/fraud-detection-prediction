import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Data Loading 

train_trans = pd.read_csv("train_transaction.csv")
train_id    = pd.read_csv('train_identity.csv')
df = train_trans.merge(train_id, on='TransactionID', how='left')
print("Shape:", df.shape)
print("\nFraud Rate:", round(df['isFraud'].mean() * 100, 2), "%")


# Data Cleaning 

missing_frac = df.isnull().mean()
cols_to_drop = missing_frac[missing_frac > 0.7].index.tolist()
df = df.drop(columns=cols_to_drop)

print("Columns dropped:", len(cols_to_drop))
print("Remaining shape:", df.shape)


# Feature Engineering 

df['SeverityTier'] = 'Medium'
df.loc[df['TransactionAmt'] < 50,  'SeverityTier'] = 'Low'
df.loc[df['TransactionAmt'] > 500, 'SeverityTier'] = 'High'

print("\nSeverity Tier counts:")
print(df['SeverityTier'].value_counts())


# Preprocessing 

x = df.drop(columns=['isFraud', 'TransactionID', 'SeverityTier'])
y = df['isFraud']

le = LabelEncoder()
for col in x.select_dtypes(include='object').columns:
    x[col] = le.fit_transform(x[col].astype(str))

x = x.fillna(x.median())

print("\nFeature matrix shape:", x.shape)


# Train Test Split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

print("Train size:", x_train.shape[0])
print("Test size:", x_test.shape[0])
print("Fraud in train:", round(y_train.mean() * 100, 2), "%")
print("Fraud in test:", round(y_test.mean() * 100, 2), "%")


# Class Imbalance 

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 2)

print("Legit transactions:", neg)
print("Fraud transactions:", pos)
print("scale_pos_weight:", spw)


# SMOTE 

sm = SMOTE(sampling_strategy=0.20, random_state=1)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)

print("\nAfter SMOTE train size:", x_train_sm.shape[0])
print("Fraud after SMOTE:", round(y_train_sm.mean() * 100, 2), "%")


# Model Building 

lr = LogisticRegression(max_iter=1000, random_state=1)
lr.fit(x_train_sm, y_train_sm)
lr_pred = lr.predict_proba(x_test)[:, 1]
print("\nLogistic Regression AUC-ROC:", round(roc_auc_score(y_test, lr_pred), 4))

rf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
rf.fit(x_train_sm, y_train_sm)
rf_pred = rf.predict_proba(x_test)[:, 1]
print("Random Forest AUC-ROC:", round(roc_auc_score(y_test, rf_pred), 4))

# XGBoost with SMOTE
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                scale_pos_weight=1, eval_metric='auc',
                                random_state=1, n_jobs=-1)
xgb_model.fit(x_train_sm, y_train_sm)
xgb_pred = xgb_model.predict_proba(x_test)[:, 1]
print("XGBoost with SMOTE AUC-ROC:", round(roc_auc_score(y_test, xgb_pred), 4))

# XGBoost with scale_pos_weight (for comparison)
xgb_spw = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              scale_pos_weight=spw, eval_metric='auc',
                              random_state=1, n_jobs=-1)
xgb_spw.fit(x_train, y_train)
xgb_spw_pred = xgb_spw.predict_proba(x_test)[:, 1]
print("XGBoost with scale_pos_weight AUC-ROC:", round(roc_auc_score(y_test, xgb_spw_pred), 4))


# Severity Tier Analysis 

x_test_copy = x_test.copy()
x_test_copy['isFraud']      = y_test.values
x_test_copy['xgb_prob']     = xgb_pred
x_test_copy['predicted']    = (x_test_copy['xgb_prob'] >= 0.5).astype(int)
x_test_copy['SeverityTier'] = df.loc[x_test.index, 'SeverityTier'].values

print("\nPerformance by Severity Tier:")
for tier in ['Low', 'Medium', 'High']:
    subset = x_test_copy[x_test_copy['SeverityTier']==tier]
    auc    = roc_auc_score(subset['isFraud'], subset['xgb_prob'])
    caught = subset[(subset['isFraud']==1) & (subset['predicted']==1)].shape[0]
    recall = caught / subset['isFraud'].sum()
    print("\n" + tier + " Severity")
    print("  Fraud cases:", subset['isFraud'].sum())
    print("  AUC-ROC:", round(auc, 4))
    print("  Recall:", round(recall, 4))


# Visualizations 

# Fraud rate by severity tier
severity_order = ['Low', 'Medium', 'High']
severity_fraud = df.groupby('SeverityTier')['isFraud'].mean() * 100
severity_fraud = severity_fraud.reindex(severity_order)

plt.figure(figsize=(6, 4))
plt.bar(severity_fraud.index, severity_fraud.values, color=['gold', 'orange', 'tomato'])
plt.title('Fraud Rate by Severity Tier')
plt.xlabel('Severity Tier')
plt.ylabel('Fraud Rate (%)')
plt.show()

# Model AUC-ROC comparison
models = ['Logistic Regression', 'Random Forest', 'XGBoost (SMOTE)', 'XGBoost (scale_pos_weight)']
scores = [round(roc_auc_score(y_test, lr_pred), 4),
          round(roc_auc_score(y_test, rf_pred), 4),
          round(roc_auc_score(y_test, xgb_pred), 4),
          round(roc_auc_score(y_test, xgb_spw_pred), 4)]

plt.figure(figsize=(8, 4))
plt.bar(models, scores, color=['steelblue', 'orange', 'tomato', 'green'])
plt.ylim(0.5, 1.0)
plt.title('Model AUC-ROC Comparison')
plt.ylabel('AUC-ROC Score')
plt.xticks(rotation=15)
plt.show()

# Top 20 feature importances
feat_imp = pd.Series(xgb_model.feature_importances_, index=x.columns)
top20    = feat_imp.nlargest(20).sort_values()

plt.figure(figsize=(8, 6))
plt.barh(top20.index, top20.values, color='steelblue')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.show()

# Transaction amount distribution
plt.figure(figsize=(8, 4))
df[df['isFraud']==0]['TransactionAmt'].clip(upper=1000).hist(bins=50, alpha=0.6, label='Legit', color='steelblue')
df[df['isFraud']==1]['TransactionAmt'].clip(upper=1000).hist(bins=50, alpha=0.6, label='Fraud', color='tomato')
plt.xlabel('Transaction Amount (capped at $1000)')
plt.ylabel('Count')
plt.title('Transaction Amount: Fraud vs Legit')
plt.legend()
plt.show()

# Fraud rate by email domain
top_domains  = df['P_emaildomain'].value_counts().head(10).index
domain_fraud = df[df['P_emaildomain'].isin(top_domains)].groupby('P_emaildomain')['isFraud'].mean() * 100
domain_fraud = domain_fraud.sort_values(ascending=False)

plt.figure(figsize=(8, 4))
plt.bar(domain_fraud.index, domain_fraud.values, color='steelblue')
plt.title('Fraud Rate by Email Domain (Top 10)')
plt.xlabel('Email Domain')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Device type if available
if 'DeviceType' in df.columns:
    device_fraud = df.groupby('DeviceType')['isFraud'].mean() * 100
    plt.figure(figsize=(5, 4))
    plt.bar(device_fraud.index, device_fraud.values, color='steelblue')
    plt.title('Fraud Rate by Device Type')
    plt.xlabel('Device Type')
    plt.ylabel('Fraud Rate (%)')
    plt.show()
else:
    print("DeviceType column not available after cleaning")

# Confusion matrix
xgb_labels = (xgb_pred >= 0.5).astype(int)
cm = confusion_matrix(y_test, xgb_labels)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nConfusion Matrix:")
print("True Negatives:", cm[0][0])
print("False Positives:", cm[0][1])
print("False Negatives:", cm[1][0])
print("True Positives:", cm[1][1])

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, xgb_pred)
f1_scores     = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx      = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print("\nRecommended Threshold:", round(best_threshold, 4))
print("Precision at threshold:", round(precision[best_idx], 4))
print("Recall at threshold:", round(recall[best_idx], 4))
print("F1-Score at threshold:", round(f1_scores[best_idx], 4))

plt.figure(figsize=(7, 4))
plt.plot(recall, precision, color='steelblue')
plt.scatter(recall[best_idx], precision[best_idx], color='tomato', s=80,
            label='Best Threshold: ' + str(round(best_threshold, 4)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBoost')
plt.legend()
plt.show()

# feature selection
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# Feature Selection - Top 15 Features from XGBoost

feat_imp = pd.Series(xgb_model.feature_importances_, index=x.columns)
top15 = feat_imp.nlargest(15)

print("Top 15 Features:")
print(top15.round(4))

x_train_top15 = x_train_sm[top15.index]
x_test_top15  = x_test[top15.index]


# V-Series Feature Analysis

v_features = feat_imp[feat_imp.index.str.startswith('V')]
v_top = v_features.nlargest(10)

print("\nTop 10 V-Series Features:")
print(v_top.round(4))

print("\nV-series in top 15 overall:")
v_in_top15 = [f for f in top15.index if f.startswith('V')]
print(v_in_top15)

plt.figure(figsize=(8, 5))
plt.barh(v_top.index, v_top.values, color='steelblue')
plt.title('Top 10 V-Series Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.show()


# XGBoost Retrained on Top 15 Features

xgb_top15 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                scale_pos_weight=1, eval_metric='auc',
                                random_state=1, n_jobs=-1)
xgb_top15.fit(x_train_top15, y_train_sm)
xgb_top15_pred = xgb_top15.predict_proba(x_test_top15)[:, 1]

print("\nXGBoost (Top 15 Features) AUC-ROC:", round(roc_auc_score(y_test, xgb_top15_pred), 4))
print("XGBoost (All Features) AUC-ROC:    ", round(roc_auc_score(y_test, xgb_pred), 4))


# Cross Validation - Final Model Evaluation

print("\nCross Validation (5-fold) on XGBoost with SMOTE:")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

cv_scores = cross_val_score(xgb_model, x_train_sm, y_train_sm,
                             cv=cv, scoring='roc_auc', n_jobs=-1)

print("Fold AUC Scores:", [round(s, 4) for s in cv_scores])
print("Mean AUC:", round(cv_scores.mean(), 4))
print("Std AUC:", round(cv_scores.std(), 4))

plt.figure(figsize=(6, 4))
plt.bar(range(1, 6), cv_scores, color='steelblue')
plt.axhline(cv_scores.mean(), color='tomato', linestyle='--',
            label='Mean AUC: ' + str(round(cv_scores.mean(), 4)))
plt.title('5-Fold Cross Validation AUC-ROC (XGBoost)')
plt.xlabel('Fold')
plt.ylabel('AUC-ROC')
plt.legend()
plt.show()


# Cost-Sensitive Threshold - Weighted by Severity

print("\nCost-Sensitive Threshold Analysis by Severity:")

x_test_copy2 = x_test.copy()
x_test_copy2['isFraud']      = y_test.values
x_test_copy2['xgb_prob']     = xgb_pred
x_test_copy2['TransactionAmt'] = df.loc[x_test.index, 'TransactionAmt'].values
x_test_copy2['SeverityTier'] = df.loc[x_test.index, 'SeverityTier'].values

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

for tier in ['Low', 'Medium', 'High']:
    subset = x_test_copy2[x_test_copy2['SeverityTier'] == tier]
    print("\n" + tier + " Severity (TransactionAmt avg: $" + str(round(subset['TransactionAmt'].mean(), 2)) + ")")
    print("  Threshold | Recall | Precision | Fraud Caught")
    for t in thresholds:
        pred_t = (subset['xgb_prob'] >= t).astype(int)
        caught = ((subset['isFraud'] == 1) & (pred_t == 1)).sum()
        total  = subset['isFraud'].sum()
        recall = caught / total if total > 0 else 0
        tp = caught
        fp = ((subset['isFraud'] == 0) & (pred_t == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        print("  ", t, "      |", round(recall, 3), "|", round(prec, 3), "    |", caught, "/", total)


# Recommended Threshold per Tier

print("\nRecommended Thresholds:")
print("  High Severity (>$500)   : 0.3  - prioritize recall, catch more fraud")
print("  Medium Severity ($50-500): 0.4  - balance precision and recall")
print("  Low Severity (<$50)     : 0.5  - default threshold is fine")


# Final Model Summary

print("\n===== FINAL MODEL SUMMARY =====")
print("Best Model: XGBoost with SMOTE")
print("AUC-ROC (full features):", round(roc_auc_score(y_test, xgb_pred), 4))
print("AUC-ROC (top 15 features):", round(roc_auc_score(y_test, xgb_top15_pred), 4))
print("Cross-Val Mean AUC:", round(cv_scores.mean(), 4))
print("Cross-Val Std:", round(cv_scores.std(), 4))
print("\nSeverity Tier Recall Summary:")
print("  Low    : 0.5033")
print("  Medium : 0.3384")
print("  High   : 0.1629  <-- most critical gap")
print("\nKey Insight: High-severity fraud recall improves significantly")
print("when threshold is lowered from 0.5 to 0.3")


# Live Interactive Fraud Prediction

print("===== FRAUD DETECTION - LIVE TRANSACTION CHECK =====")
print("Enter transaction details below:\n")

amt        = float(input("Transaction Amount ($): "))
card_type  = input("Card Type (credit/debit): ").strip().lower()
email      = input("Purchaser Email Domain (e.g. gmail.com): ").strip().lower()
device     = input("Device Type (desktop/mobile): ").strip().lower()
c1         = float(input("Number of addresses linked to card (e.g. 1): "))
c8         = float(input("Number of days since last transaction (e.g. 30): "))

# build a blank transaction using median values from training
new_txn = pd.DataFrame([x_train_sm.median()], columns=x_train_sm.columns)

# fill in what the user provided
new_txn['TransactionAmt'] = amt

if 'C1' in new_txn.columns:
    new_txn['C1'] = c1
if 'C8' in new_txn.columns:
    new_txn['C8'] = c8

# encode card type
if 'card4' in new_txn.columns:
    new_txn['card4'] = 1 if card_type == 'credit' else 0

# assign severity tier
if amt < 50:
    tier = 'Low'
    threshold = 0.5
elif amt <= 500:
    tier = 'Medium'
    threshold = 0.4
else:
    tier = 'High'
    threshold = 0.2

# predict
prob      = xgb_model.predict_proba(new_txn)[0][1]
predicted = 1 if prob >= threshold else 0

print("\n===== PREDICTION RESULT =====")
print("Transaction Amount :", "$" + str(round(amt, 2)))
print("Severity Tier      :", tier)
print("Threshold Applied  :", threshold)
print("Fraud Probability  :", round(prob * 100, 2), "%")

if predicted == 1:
    print("Decision           : FRAUD - Transaction Flagged")
else:
    print("Decision           : LEGIT - Transaction Approved")
    
    
    
## live prediction from test data
# Live Interactive Fraud Prediction

print("===== FRAUD DETECTION - LIVE TRANSACTION CHECK =====")
print("Choose mode:\n")
print("1 - Enter a real transaction ID from the dataset")
print("2 - Manual input (limited accuracy)")

mode = input("\nEnter 1 or 2: ").strip()

if mode == '1':
    print("\nSample FRAUD transaction IDs to try:")
    fraud_samples = x_test[y_test == 1].index[:5].tolist()
    for idx in fraud_samples:
        print("  TransactionID:", idx, "| Amount: $" + str(round(df.loc[idx, 'TransactionAmt'], 2)),
              "| Tier:", df.loc[idx, 'SeverityTier'])

    print("\nSample LEGIT transaction IDs to try:")
    legit_samples = x_test[y_test == 0].index[:5].tolist()
    for idx in legit_samples:
        print("  TransactionID:", idx, "| Amount: $" + str(round(df.loc[idx, 'TransactionAmt'], 2)),
              "| Tier:", df.loc[idx, 'SeverityTier'])

    txn_id = int(input("\nEnter Transaction ID: ").strip())

    if txn_id not in x_test.index:
        print("Transaction ID not found in test set.")
    else:
        new_txn = x_test.loc[[txn_id]]
        amt     = df.loc[txn_id, 'TransactionAmt']
        tier    = df.loc[txn_id, 'SeverityTier']
        actual  = y_test.loc[txn_id]

        if tier == 'High':
            threshold = 0.2
        elif tier == 'Medium':
            threshold = 0.4
        else:
            threshold = 0.5

        prob      = xgb_model.predict_proba(new_txn)[0][1]
        predicted = 1 if prob >= threshold else 0

        print("\n===== PREDICTION RESULT =====")
        print("Transaction ID     :", txn_id)
        print("Transaction Amount : $" + str(round(amt, 2)))
        print("Severity Tier      :", tier)
        print("Threshold Applied  :", threshold)
        print("Fraud Probability  :", round(prob * 100, 2), "%")

        if predicted == 1:
            print("Decision           : FRAUD - Transaction Flagged")
        else:
            print("Decision           : LEGIT - Transaction Approved")

        print("Actual Label       :", "FRAUD" if actual == 1 else "LEGIT")
        print("Result             :", "CORRECT" if predicted == actual else "WRONG")

else:
    print("\nEnter transaction details below:\n")

    amt       = float(input("Transaction Amount ($): "))
    c1        = float(input("Number of addresses linked to card (e.g. 1): "))
    c8        = float(input("Number of days since last transaction (e.g. 30): "))

    new_txn = pd.DataFrame([x_train_sm.median()], columns=x_train_sm.columns)
    new_txn['TransactionAmt'] = amt
    if 'C1' in new_txn.columns:
        new_txn['C1'] = c1
    if 'C8' in new_txn.columns:
        new_txn['C8'] = c8

    if amt < 50:
        tier = 'Low'
        threshold = 0.5
    elif amt <= 500:
        tier = 'Medium'
        threshold = 0.4
    else:
        tier = 'High'
        threshold = 0.2

    prob      = xgb_model.predict_proba(new_txn)[0][1]
    predicted = 1 if prob >= threshold else 0

    print("\n===== PREDICTION RESULT =====")
    print("Transaction Amount : $" + str(round(amt, 2)))
    print("Severity Tier      :", tier)
    print("Threshold Applied  :", threshold)
    print("Fraud Probability  :", round(prob * 100, 2), "%")

    if predicted == 1:
        print("Decision           : FRAUD - Transaction Flagged")
    else:
        print("Decision           : LEGIT - Transaction Approved")

    print("\nNote: Manual mode uses median values for V-series features.")
    print("For accurate predictions use Mode 1 with a real Transaction ID.")