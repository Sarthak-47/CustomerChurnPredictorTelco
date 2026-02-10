# Telco Customer Churn Prediction

## Project Overview
This repository contains an end-to-end machine learning pipeline for predicting customer churn in the telecommunications domain using structured customer and service usage data. The project focuses on clean preprocessing, feature engineering, multiple model training, and automated model comparison.

---

## Dataset

- **Dataset Name:** IBM Telco Customer Churn Dataset
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target Variable:** `Churn`  
  - 1 → Customer churned  
  - 0 → Customer did not churn

The dataset includes customer demographics, subscribed services and billing information.

---

## Train / Test Split

- **Method:** `train_test_split`
- **Split Ratio:** 80% train / 20% test
- **Random State:** 42
- **Stratification:** Applied on the churn label

---

## Models Trained

The following models were trained on the same preprocessing pipeline and data split:

- K-Nearest Neighbours
- Support Vector Classifier (SVC)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- Voting Classifier (ensemble)

---

## Evaluation Metrics

- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

---

## Final Results (Test Set Accuracy)

| Model                          | Variable Name | Accuracy |
|--------------------------------|---------------|----------|
| K-Nearest Neighbours           | knn_model     | 0.7758   |
| Support Vector Classifier      | svc_model     | 0.8076   |
| Logistic Regression            | lr_model      | 0.8090   |
| Decision Tree                  | dt_model      | 0.7289   |
| Random Forest                  | model_rf      | 0.8137   |
| AdaBoost                       | a_model       | 0.8128   |
| Gradient Boosting              | gb            | 0.8081   |
| Voting Classifier (Ensemble)   | eclf1         | 0.8161   |

---

## Best Model

The **Voting Classifier (ensemble)** achieved the best performance on the test set with:

**Accuracy = 0.8161**

This model combines the predictions of multiple base learners, leading to improved generalisation compared to individual models.

---

## Key Observations

- Ensemble methods (Random Forest, AdaBoost, Gradient Boosting and Voting Classifier) outperform most single models.
- Decision Tree shows the weakest performance, indicating high variance and overfitting.
- Combining diverse classifiers using a voting strategy provides the strongest overall accuracy.

---

## How to Run

1. Place the dataset file

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

in the same directory as the notebook.

2. Open and run:

```
customer_churn_predictor_with_inline_markdown.ipynb
```

3. Execute all cells from top to bottom to reproduce the results.

---

## Tools and Libraries

- Python
- pandas, numpy
- scikit-learn
- matplotlib

---

## Conclusion

This project demonstrates a reproducible and extensible churn prediction pipeline. By systematically comparing multiple classifiers and ensemble methods on the same processed dataset, the study shows that ensemble learning, particularly voting-based approaches, provides the most reliable performance for this churn prediction task.
