from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#import xgboost with if handle missing
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # Handle the case where xgboost is not installed

class ModelFactory:
    @staticmethod
    def get_model():
        models = {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "bayesian_ridge": BayesianRidge(max_iter=1000),
        }
        if XGBRegressor:
            models["xgboost"] = XGBRegressor(
                n_estimators=1000, #due to large dataset
                learning_rate=0.05, #Frce the model to learn slowly
                max_depth=7, #superconductivity is non linear a deeper tree allows model to learn more specific chemical combinations
                subsample=0.8, #trains on random 80% of data for each tree to prevent model from memorizing specific data points
                colsample_bytree=0.8, #due to columns have similar values, it prevents model from lazily taking 2-3 strong features. this forces it to learn from a wider variety of features
                n_jobs=-1, #makes model use all cpu cores cutting down processing time
                random_state=42,
            )
        return models