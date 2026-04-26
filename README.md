# IEEE-CIS Fraud Detection

> End-to-end fraud detection pipeline on 590,000+ transactions — achieving 0.926 AUC-ROC using XGBoost with SMOTE-based class imbalance handling and severity segmentation.

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-6DB33F?style=flat)

---

## 📌 Business Problem

Financial fraud costs billions annually — and traditional rule-based systems struggle to keep pace with evolving fraud patterns. This project builds a **machine learning pipeline that flags fraudulent transactions in real time**, with a focus on high-severity cases involving large transaction amounts.

Key questions:
- Can we accurately distinguish fraudulent from legitimate transactions at scale?
- How do we handle extreme class imbalance (fraud is rare by nature)?
- Which transaction features are the strongest fraud signals?

---

## 📊 Data

| Attribute | Detail |
|---|---|
| Dataset | IEEE-CIS Fraud Detection (Kaggle)| 
| Total transactions | 590,000+ |
| Fraud rate | ~3.5% (heavily imbalanced) |
| Key fields | Transaction amount, card type, device, email domain, time delta, geography |

---

## ⚙️ Methodology

1. **Data Cleaning & Merging** — Joined transaction and identity tables; handled 400+ features with high missingness rates
2. **Exploratory Analysis** — Identified fraud patterns by transaction amount, geography, device type, and time of day
3. **Severity Segmentation** — Categorized transactions into high/medium/low severity tiers based on transaction amount and geographic risk to prioritize model attention
4. **Feature Engineering** — Encoded categorical variables, created interaction features, handled missing values via median/mode imputation
5. **Class Imbalance Handling** — Applied SMOTE (Synthetic Minority Oversampling Technique) to address the 96.5%/3.5% class split
6. **Model Development** — Trained and compared Logistic Regression, Random Forest, and XGBoost; tuned hyperparameters via cross-validation
7. **Evaluation** — Assessed on AUC-ROC, precision, recall, and F1 — prioritizing recall to minimize missed fraud cases

---

## 📈 Key Results

| Metric | Result |
|---|---|
| AUC-ROC | **0.926** |
| Best Model | XGBoost |
| Transactions analyzed | **590,000+** |
| Class imbalance method | SMOTE |
| Differentiator | Severity segmentation by transaction amount and geography |

---

## 🖼️ Screenshots / Visuals



---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/SaloniShahi336/fraud-detection-prediction.git
cd fraud-detection-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/fraud_detection_modeling.ipynb
```

**Folder structure:**

├── data/               # Raw and processed datasets

├── notebooks/          # Modeling notebook

├── src/                # Pipeline scripts

└── requirements.txt

> **Data source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)

---

## 💡 Learnings

- SMOTE must be applied only on training data — applying it before the train/test split causes data leakage and inflates performance metrics
- XGBoost's `scale_pos_weight` parameter is an alternative to SMOTE that's worth comparing — it handles imbalance differently and can outperform oversampling on certain distributions
- Severity segmentation added real business value: flagging high-value fraudulent transactions first aligns model output with actual financial risk, not just accuracy scores
- With 400+ features, feature selection matters as much as model choice — removing low-signal features improved both speed and generalization
