# Customer-Churn-Prediction

## Description
A data science and machine learning project in predicting customer churn using supervised learning algorithms including AdaBoost and other models.

## Overview of Key Steps
- **Exploratory Data Analysis (EDA)**
  - Identified key variables associated with customer churn, including:
     - Contract type.
     - Tenure (period of commitment).
     - Internet service type.
- **Preprocessing**
  - Addressed class imbalance with class weights and/or SMOTE.
  - Encoded categorical variables and scaled features where appropriate.
- **Model Training**
  - Compared various classification algorithms, including Logistic Regression, Support Vector Machines and Ensemble Methods.
  - Selected AdaBoost as the final model based on strong recall of 87% on churn class.
 
## Tools, Libraries and Skills
- **Programming Language:** Python
- **Libraries:** pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, imbalanced-learn, io, joblib
- **Environment:** Jupyter Notebook
- **Skills Used:**
  - Data Visualisation
  - Feature Engineering
  - Inferential Statistics
  - Machine Learning
  - Programming
  - Predictive Modelling
 
## Results
- **Final model:** AdaBoost
- **Performance Metrics**:
  - **Precision:** 50%
  - **Recall:** 87%
  - **F1-score:** 63%
  - **Average Precision (AP):** 66% 

## How to Use
### Description of Project files:
   - **Customer-Churn-Prediction.pdf:**
     A read-only version of the notebook file.
   - **Customer-Churn-Prediction.ipynb:**
     Main project file. Open in Jupyter Notebooks to explore set-up and interact with code directly.
   - **Telco-Customer-Churn-24.csv:**
     Dataset used in this project.
   - **customer_churn_predictor.pkl:**
     Pickle file of the final (AdaBoost) model, ready for deployment.
   - **customer_churn_column_names.pkl:**
     Pickle file containing feature names used by the final model.
   - **requirements.txt:**
     List of all libraries required and their versions for installation.
   - **Optional graphics:**
        - **adaboost_holdout_cmatrix.png:**
         Final model's confusion matrix.
        - **adaboost_holdout_report.png:**
         Final model's classification report.
        - **adaboost_roc_pr_curves.png:**
        Final model's ROC and Precision-Recall Curves.
### Usage
1. **Clone this repository:**
   - See [this guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for help on how to do this.
2. **Install Jupyter Notebook (if not already installed):**
   - See [this guide](https://jupyter.org/install) for help on how to do this.
3. **Install Required Libraries:**
   - See 'requirements.txt' for list of libraries and their corresponding versions.
   - For an easy set-up, open a terminal (or command prompt) and run ```pip install -r requirements.txt```.
4. **Run the Notebook:**
   - Navigate to project folder in Jupyter Notebook and open 'Customer-Churn-Prediction.ipynb'.
   - **Note:** Ensure that the Telco Customer Churn **dataset** ('Telco-Customer-Churn-24.csv') is **in the same folder** as the notebook file to avoid file path errors when running the code within the notebook.
   - Optional: See accompanying PDF for read-only version of notebook file.
5. **Explore the Notebook:**
   - Open 'Customer-Churn-Prediction.ipynb' in Jupyter Notebook and follow the cells from top to bottom.
   - **Optional**: Quickly navigate to a particular section using the Table of Contents in the Left Sidebar (View header, Left Sidebar...).
   - **Note:** Some computationally intensive code (e.g. grid searches) have been commented out. To re-run these sections, you may uncomment the relevant lines in the notebook but first ensure that your system has the computational resources available to run the code.
6.  **(Optional) Use the Pre-trained Model:**
   - To load the final model ('customer_churn_predictor.pkl'):
     ```python
     import joblib

     # Load model
     loaded_model = joblib.load("customer_churn_predictor.pkl")

     # Load column names
     loaded_columns = joblib.load("customer_churn_column_names.pkl")
     ```

## Future Work
- **Clean up the project**
   Refine the notebook and PDF by fixing typos and removing redundant code.
- **Build an API for deployment**
   Develop a REST API using Flask or Django to deploy the trained model to enable real-time predictions. Explore integrating the model within a cloud environment, such as AWS or Azure.
- **Feature pruning**
   Assess whether removing less important features can maintain or improve model performance metrics (e.g. precision, recall), thereby reducing redundancy and improving efficiency.
- **Expand hyperparameter tuning**
   Perform a more extensive hyperparameter search across the other models, particularly the ensemble methods like Random Forest and Gradient Boosting. This includes testing additional parameters and increasing the number of k-folds in cross-validation to improve model generalisation.
- **Explore other models**
   Experiment with alternative models, such as XGBoost, and compare their performance against the current final model.

## Acknowledgements
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Libraries:** pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, imbalanced-learn, io, joblib
- **Tools:** Jupyter Notebook, Python 3.11.7
- **Support:** Generative AI tools, Stack Overflow, and official library documentation were used to support troubleshooting code issues and refining parts of the write-up. All analysis, coding and modelling were performed independently.
- **Personal Note:** This project was developed over several months alongside full-time university studies and part-time work. It reflects both my technical skills and my commitment to learning and improving in the field of data science and machine learning.
