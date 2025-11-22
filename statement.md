# Project Statement: SuperConductAI

## 1. Problem Statement
Finding high-temperature superconductors is hard. Traditional methods to find the critical temperature ($T_c$) take a long time and are expensive. Right now, there isn't a simple formula that connects a material's chemicals to its $T_c$. This slows down research in science.

## 2. Scope of the Project
The aim of this repository is to demonstrate the application of machine learning in science by connecting chemical features to physical properties.

* **What this project does:**
    * Builds a regression model using 81 different atomic features.
    * Compares linear models (for ex: Linear Regression) against non-linear ones (for ex: Random Forest).
    * Creates a command-line tool to train and check the models.

* **What this project does not do:**
    * It does not have a website interface (it runs in the terminal).
    * It does not say "Yes" or "No" if something is a superconductor; it only predicts the exact temperature.

## 3. Target Users
* **Scientists:** To help them decide which materials to make in the lab.
* **Students:** To learn how atomic features (like mass and radius) affect critical temperature.

## 4. High-Level Features
* **Data Handling:** Handles missing data and splits the dataset into training and testing parts.
* **Better Learning:** Uses Random Forest and XGBoost to understand complex chemical data.

* **Results:** Prints out the $R^2$ score and MSE so we know if the model is reliable.
