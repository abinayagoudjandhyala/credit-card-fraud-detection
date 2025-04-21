# Credit Card Fraud Detection 🕵️‍♂️💳

This project tackles the problem of detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, making fraud detection a non-trivial task that requires careful preprocessing and model evaluation.

## 📂 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Features:** 30 total (28 anonymized PCA components + `Time` + `Amount`)
- **Target Variable:**
  - `0` → Legitimate
  - `1` → Fraudulent

## 🎯 Objective

To build a model that identifies fraudulent transactions with high accuracy while accounting for extreme class imbalance.

## 🔄 Workflow Summary

### 📊 Data Exploration
- Loaded dataset using Pandas
- Checked for null values
- Analyzed class distribution (highly imbalanced)
- Separated legitimate and fraudulent transactions for comparative statistics

### 🧹 Preprocessing
- No missing data treatment required
- Stratified sampling of legitimate transactions to balance the dataset
- Used `train_test_split` to divide data into training and test sets

### 🧠 Model Used
- **Logistic Regression** as the baseline model

### 📈 Evaluation Metrics
- **Accuracy Score**

> ⚠️ Note: No advanced metrics like AUC, Precision, or Recall were used yet, which are critical in fraud detection tasks.

## 🛠️ Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas scikit-learn
```

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place the CSV in the appropriate directory (`/content/creditcard.csv` or update path in the notebook)
3. Run the notebook:

```bash
jupyter notebook CreditCardFraudDetection.ipynb
```
## 📚 References

- [Original Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
