# Titanic Survival Prediction

## 1. Project Overview
This project predicts which passengers survived the Titanic disaster using a clean and robust machine learning workflow. It covers data cleaning, feature engineering, model training, evaluation, and interpretation using Python and Jupyter Notebook.

## 2. Dataset
- **Description**: Passenger data from the Titanic including age, sex, class, fare, etc.
- **Source**: A small CSV (~61 KB) downloaded directly from GitHub.
- **Key columns**:
  - `Survived`: Target variable (0 = No, 1 = Yes)
  - `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `Cabin`, and more.

## 3. Main Steps :
1. **Data Cleaning**:
   - Filled missing `Age` using median based on passenger class and gender.  
   - Imputed missing `Embarked` with the most common port.  
   - Transformed `Cabin` into `Deck` (first character) and marked missing as `MISSING`.  
   - Created flags (`Age_missing_flag`, `Embarked_missing_flag`) to note original missing data.

2. **Feature Engineering**:
   - `FamilySize = SibSp + Parch + 1`  
   - `IsAlone = 1` if passenger was alone, otherwise `0`  
   - Extracted `Title` from passenger names (e.g., "Mr", "Mrs", "Master"), grouped rare titles into “Rare”.  
   - Treated `Pclass` as categorical.

3. **Model Pipeline**:
   - Used `ColumnTransformer` to process numeric and categorical data separately:  
     - Numeric: imputation + scaling.  
     - Categorical: imputation + one-hot encoding.  
   - Final model: Logistic Regression (`liblinear`, `max_iter=1000`).

4. **Evaluation**:
   - Achieved **Accuracy ≈ 84%** on test set.  
   - **ROC-AUC ≈ 0.87**, indicating strong model ranking capability.

5. **Feature Insights**:
   - Strong positive predictors: `Sex_female`, `Title_Master` (children), `Pclass_1`.  
   - Negative predictors: `Title_Mr`, `Deck_MISSING` (unknown cabin).  

## 4. Results :
- **Classification performance**:  
  | Metric       | Value |
  | Accuracy     | 0.84  |
  | ROC-AUC      | 0.87  |

- **Model Interpretation**: A logistic regression model shows that being female or a child significantly increases survival odds, while missing cabin data or being a grown male reduces survival likelihood.

## 5. Future Improvements :
Add Survival Analysis (e.g. Kaplan-Meier).
Try advanced classifiers (Random Forest, XGBoost).
Hyperparameter tuning.

Build an interactive dashboard with Plotly or Streamlit

