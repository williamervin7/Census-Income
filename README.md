# Income Classification Based on Census Data

## 📖 Project Overview

The goal of this project was to predict whether an individual's annual income exceeds \$50K using demographic and employment-related features from the U.S. Census dataset. This is a classic binary classification problem that also includes challenges like missing values, categorical variables, and class imbalance.

## 💡 Business Objective

Build an interpretable, well-performing model to support targeted outreach or program eligibility decisions based on income classification, with an emphasis on minimizing false negatives (high earners misclassified as low earners).

## 🧼 Data Cleaning and Feature Engineering

- Removed duplicate and irrelevant columns.
- Handled missing and ambiguous entries (e.g., replacing `'?'` in categorical variables).
- Combined `capital_gain` and `capital_loss` into a new feature: `net_capital`.
- One-hot encoded all categorical features.
- Scaled numerical features using `StandardScaler`.

### Final Feature Set Used:
- **Numerical:** `education_num`, `capital_gain`, `net_capital`
- **Categorical:** `workclass`, `marital_status`, `occupation`

## ⚖️ Handling Class Imbalance

The dataset was moderately imbalanced (~76% class 0, ~24% class 1). Rather than using synthetic sampling (SMOTE, ADASYN), the models that supported it were trained using `class_weight='balanced'`.

Key evaluation metrics focused on:
- **Recall** (ability to identify true positives),
- **F1-score** (balance of precision and recall),
- **ROC AUC** (discriminatory power across thresholds).

## 🤖 Models Evaluated

| Model                | F1 Score | ROC AUC | Accuracy | Recall |
|---------------------|----------|---------|----------|--------|
| Logistic Regression | 0.674    | 0.901   | 0.805    | 0.834  |
| Random Forest       | 0.667    | 0.895   | 0.846    | 0.640  |
| Gradient Boosting   | 0.675    | 0.918   | 0.862    | 0.593  |
| **XGBoost**         | 0.703    | 0.926   | 0.870    | 0.637  |
| **LightGBM** ✅      | **0.708**| **0.926**| **0.828**| **0.865**|

## 🏆 Best Model: LightGBM

LightGBM consistently outperformed other models, especially in terms of **recall** and **ROC AUC**, which were critical for this problem. Its ability to handle categorical features, missing data, and large datasets efficiently made it an ideal choice.

## 🔍 Hyperparameter Tuning

GridSearchCV was used to tune hyperparameters of the LightGBM model with `scoring={'f1', 'roc_auc', 'accuracy', 'recall'}` and `refit='recall'`.

**Best Parameters:**
```python
{
  'learning_rate': 0.1,
  'max_depth': 5,
  'n_estimators': 800,
  'num_leaves': 63
}
```
## 📊 Train/Test Evaluation
After fitting the final LightGBM pipeline on the full training data, the model was evaluated on both the training and original test set
The model performs consistently across both training and test data, with no major signs of overfitting.

* Accuracy holds steady at ~82–83%, which is solid given the complexity of the task.
* The model achieves high recall on the >50K class (~87–90%), meaning it successfully captures most high-income individuals.
* Precision for >50K is lower (~58–59%), indicating some false positives. This is a common tradeoff with imbalanced data.
* Overall, the model generalizes well and balances performance across both classes.

📦 Libraries Used
pandas, numpy
scikit-learn
xgboost
lightgbm
matplotlib, seaborn

✅ Key Takeaways
- LightGBM provided the best performance for this problem with the best balance of recall and overall accuracy.
- Class imbalance was effectively handled through class_weight='balanced', avoiding the need for oversampling.
- The project demonstrates the full machine learning workflow: cleaning, preprocessing, model building, evaluation, and tuning — with real-world messy data.

📁 Future Improvements
- Explore SHAP or permutation importance for deeper model interpretability.
- Deploy model using Streamlit or Flask to make predictions interactively.
- Try ensemble stacking using LightGBM + Logistic Regression to further improve performance.
