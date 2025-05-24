# 💼 Bank Marketing Prediction using Decision Tree

This project uses a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit, based on the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) from UCI Machine Learning Repository.

---

## 📌 Objective

To train a Decision Tree model that predicts whether a customer will respond positively (`yes`) or negatively (`no`) to a marketing campaign conducted by a Portuguese bank.

---

## 🗂️ Dataset Details

- **Source**: UCI ML Repository  
- **File**: `bank-full.csv`  
- **Separator**: Semicolon (`;`)

**Target Variable:**  
- `y` — Whether the client subscribed to a term deposit (binary: `yes`, `no`)

---

## 🛠️ Tools & Libraries

- Python 3
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## ⚙️ Workflow

1. **Data Loading**: Reads the dataset from CSV format.
2. **Data Preprocessing**:
   - Encodes categorical variables using `LabelEncoder`.
   - Splits the dataset into training and test sets.
3. **Model Training**:
   - Uses a `DecisionTreeClassifier` with `max_depth=5`.
4. **Model Evaluation**:
   - Outputs accuracy, confusion matrix, and classification report.
5. **Visualization**:
   - Plots the trained decision tree using `plot_tree()` from `sklearn`.

---

## 📊 Sample Output

- **Classification Report**
- **Confusion Matrix**
- **Model Accuracy**
- **Visualized Decision Tree**

---
