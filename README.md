# SuperConductAI: Critical Temperature prediction

## 1. Project Title: 
**SuperConductAI: Machine Learning for Superconductor Critical Temperature Prediction**

## 2. Overview
SuperConductAI is a machine learning pipeline designed to predict the critical temperature ($T_c$) of superconducting materials based on their chemical formula and thermodynamic properties.
By leveraging ensemble regression techniques (Random Forest, XGBoost), this tool allows researchers to virtually screen materials, significantly reducing the cost and time associated with
experimental synthesis.

## 3 Features:
* **Data Loading:** Automatically loads and cleans the dataset, which is quite long (81 columns).
* **Multiple Models:** Trains several models at once: Linear Regression, Bayesian Ridge, Decision Tree, Random Forest, and XGBoost.
* **Evaluation:** Calculates accuracy scores like MSE and $R^2$ i.e squared correlation coefficient to see how well the models work.
* **Visualization:** Creates scatter plots to show the difference between the actual values and the predicted values.
* **Organized Code:** The code is split into different files (Data, Logic, Main) to keep it clean.

## 4. Technologies Used
* **Language:** Python 3.8+
* **Data Tools:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Graphs:** Matplotlib, Seaborn
* **Version Control:** Git & GitHub

## 5. How to Install & Run
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Phinnaeus007/SuperConductAI.git
    cd SuperConductAI
    ```

2.  **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Code:**
    ```bash
    python main.py
    ```

## 6. How to Test
To check if the data loading works correctly, run this command:
```bash
python tests/test_data_loader.py
```
## Precaution:
Make sure your directory is SuperConductAI before running
```bash
python test/test_data_loader.py
```
