# Telco Customer Churn Prediction

## Project Overview
This project focuses on predicting **customer churn** in the telecommunications domain using supervised machine learning models.
Churn prediction helps telecom providers identify customers who are likely to discontinue services and enables proactive retention strategies.

The problem is framed as a **binary classification task**, where the objective is to predict whether a customer will churn or not.

This project is implemented as a fully reproducible Jupyter notebook with inline explanations and automatic model comparison.

---

## Dataset

- **Dataset Name:** IBM Telco Customer Churn Dataset
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target Variable:** `Churn`
  - `1` → Customer churned
  - `0` → Customer did not churn

The dataset contains customer demographics, subscribed services and billing information.

Typical features include:

- customer tenure
- monthly charges and total charges
- internet and phone services
- online security, backup and technical support services
- streaming services
- contract type and payment method
- billing preferences

---

## Train / Validation Split Method

- **Splitting Technique:** `train_test_split`
- **Train / Test Ratio:** 80% / 20%
- **Random State:** 42
- **Stratification:** Applied on the churn label

The test set is treated as unseen data and is used exclusively for model evaluation.

---

## Models Trained

The following machine learning models were trained and evaluated using the same preprocessing pipeline and the same data split:

1. **Random Forest Classifier**
2. **AdaBoost Classifier**

The notebook design also allows additional models to be trained and automatically included in the comparison stage without modifying the evaluation logic.

---

## Evaluation Metrics

The models were evaluated using multiple metrics to reflect both overall performance and churn detection quality.

### Metrics reported

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Model Performance Summary

The notebook prints the complete classification report and confusion matrix for each trained model.
Below is the reporting format used for every classifier.

### Random Forest Classifier

- **Accuracy:** Reported in notebook output
- **Recall (Churn):** Reported in notebook output
- **F1-score (Churn):** Reported in notebook output

Confusion Matrix:
```
[[TN  FP]
 [FN  TP]]
```

---

### AdaBoost Classifier

- **Accuracy:** Reported in notebook output
- **Recall (Churn):** Reported in notebook output
- **F1-score (Churn):** Reported in notebook output

Confusion Matrix:
```
[[TN  FP]
 [FN  TP]]
```

---

## Model Comparison Grid

All trained models are evaluated on the same test set and summarised using the automatic comparison cell.

| Model         | Accuracy |
|---------------|----------|
| Random Forest | Reported in notebook |
| AdaBoost      | Reported in notebook |

The comparison table and the corresponding bar plot are generated automatically at runtime.

---

## Best Model Selection

The best-performing model is selected based on the **highest test accuracy** produced by the final comparison cell.

For business-oriented churn prediction, additional emphasis should be placed on:

- recall for the churn class
- F1-score

These metrics are reported for every model inside the notebook.

---

## Key Insights

- Proper preprocessing of billing features such as `TotalCharges` significantly improves model stability.
- Normalising service-related categories (for example merging “No internet service” into “No”) reduces unnecessary feature sparsity.
- Feature engineering using spending and tenure information improves predictive performance.
- Ensemble-based models such as Random Forest and AdaBoost capture non-linear interactions between customer behaviour and churn more effectively than simple linear baselines.
- Accuracy alone is not sufficient to judge churn models; recall and F1-score provide more realistic insight into churn detection capability.

---

## Reproducibility

- Fixed random seed (`random_state = 42`)
- Identical preprocessing pipeline for all models
- Identical train–test split for all models
- Consistent evaluation methodology

---

## Tools and Libraries

- Python
- pandas
- numpy
- scikit-learn
- matplotlib

---

## How to Run

1. Place the dataset file

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

in the same directory as the notebook.

2. Open the notebook

```
customer_churn_predictor_with_inline_markdown.ipynb
```

3. Run all cells from top to bottom.

The final cells will automatically:

- evaluate all trained models
- print their metrics
- generate the comparison plot

---

## Conclusion

This project demonstrates a structured and reproducible machine learning pipeline for telecom churn prediction.
By combining careful data cleaning, domain-driven feature engineering and ensemble learning methods, the system provides a reliable framework for identifying customers at risk of churn and supports data-driven retention strategies.
