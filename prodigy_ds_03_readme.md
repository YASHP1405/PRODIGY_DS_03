# PRODIGY_DS_03 – Decision Tree Classifier (Bank Marketing Dataset)

## 📌 Project Overview
This project is part of the **Prodigy InfoTech Data Science Internship – Task 03**. The objective is to build a **Decision Tree Classifier** to predict whether a customer will purchase a product (term deposit) based on their **demographic and behavioral data**.

The **Bank Marketing Dataset** from the **UCI Machine Learning Repository** is used for this task.

---

## 📂 Dataset Description
The dataset consists of the following files:

- **bank.csv** → Primary dataset used for training and testing the model
- **bank-full.csv** → Complete dataset (optional, not used here)
- **bank-names.txt** → Metadata explaining dataset features

### 🎯 Target Variable
- **y** → Indicates whether the customer subscribed to a term deposit (`yes` or `no`)

---

## 🧰 Technologies Used
- **Python 3.11**
- **Pandas** – Data loading and preprocessing
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning model building and evaluation

---

## 🔄 Workflow
1. Load and inspect the dataset
2. Encode categorical variables
3. Split data into training and testing sets
4. Train a Decision Tree Classifier
5. Evaluate model performance
6. Visualize decision tree and results

---

## 📊 Model Evaluation
The model is evaluated using:
- Accuracy score
- Classification report (Precision, Recall, F1-score)
- Confusion matrix visualization

---

## 🔍 Key Insights
- Decision Tree effectively predicts customer purchase behavior
- Behavioral features like campaign and duration influence predictions
- The model is interpretable and suitable for business decision-making

---

## 📁 Project Structure
```
PRODIGY_DS_03/
│
├── bank.csv
├── bank-full.csv
├── bank-names.txt
├── Task - 03.py
└── README.md
```

---

## ▶ How to Run the Project
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/PRODIGY_DS_03.git
   ```
2. Navigate to the project directory
   ```bash
   cd PRODIGY_DS_03
   ```
3. Install required libraries
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the script
   ```bash
   python "Task - 03.py"
   ```

---

## ✅ Conclusion
This project demonstrates how a Decision Tree classifier can be used to predict customer purchasing behavior using demographic and behavioral features. The model provides clear decision rules, making it useful for targeted marketing strategies.

---

## 📬 Author
**Yash Pawar**  
Data Science Intern – Prodigy InfoTech

---

⭐ If you find this project useful, consider starring the repository!