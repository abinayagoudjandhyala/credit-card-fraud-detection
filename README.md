# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using a logistic regression model. This project leverages real-world anonymized data from European cardholders in September 2013.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Frauds**: 492 (~0.17%)
- **Features**: 30
  - `V1` to `V28`: Anonymized via PCA
  - `Time` and `Amount`: Original features
  - `Class`: Target (0 = Legit, 1 = Fraud)

---

## ⚙️ Technologies Used

- Python (Jupyter Notebook)
- NumPy, Pandas
- Scikit-learn
- Logistic Regression

---

## 🧠 Workflow

### 1. **Data Exploration**
- Checked for missing values (none)
- Separated legitimate and fraudulent transactions
- Visualized value distributions and stats

### 2. **Data Balancing**
- Random undersampling of legitimate transactions to match the number of frauds
- Combined for a balanced dataset with equal fraud and legit samples

### 3. **Modeling**
- Model: `LogisticRegression`
- Training/Test Split: 80/20 using stratified sampling
- Accuracy evaluated on both training and test sets

---

## 📈 Results

| Metric              | Score     |
|---------------------|-----------|
| Training Accuracy   | ~94%      |
| Test Accuracy       | ~91%      |
| Balanced Dataset    | 206 rows  |

⚠️ Note: Accuracy is not the best metric for imbalanced datasets — future improvements can involve precision, recall, F1-score, and ROC-AUC.

---

## 🔮 Predicting New Transactions

To use the model for a custom transaction:

```python
import numpy as np

# Replace with 30 feature values from a transaction
input_data = (0.1, -0.2, ..., 1.23)  # 30 values

input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_as_numpy_array)

if prediction[0] == 1:
    print("⚠️ Fraudulent transaction detected.")
else:
    print("✅ Legitimate transaction.")
```

---

## 🚀 Future Improvements

- Try more powerful models: Random Forest, XGBoost, SVM
- Handle class imbalance with SMOTE or ADASYN
- Visualize ROC curves and confusion matrices
- Deploy as a web app or REST API

---

## 🧑‍💻 Author

Abinaya Goud
abinayagoud23@gmail.com

---

## 📜 License

This project is licensed under the MIT License.


