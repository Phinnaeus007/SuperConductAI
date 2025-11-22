import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator

class Evaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, model: BaseEstimator, name, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        
        self.results[name] = {'MSE': mse, 'R2': r2, 'MAE': mae, 'pred': pred}
        return self.results[name]

    def print_summary(self):
        print("\n" + "="*40)
        print(f"{'Model Name':<20} | {'MSE':<10} | {'R2':<10}")
        print("-" * 45)
        for name, metrics in self.results.items():
            print(f"{name:<20} | {metrics['MSE']:.2f}       | {metrics['R2']:.4f}")
        print("="*40 + "\n")

    def plot_model_results(self, y_test, model_name):
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return

        y_pred = self.results[model_name]['pred']
        errors = np.abs(y_test - y_pred)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=y_test, y=y_pred, hue=errors, 
            palette='RdYlGn_r', edgecolor='black', linewidth=0.5, s=60
        )
        
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Critical Temp (K)')
        plt.ylabel('Predicted Critical Temp (K)')
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.legend(title="Error Magnitude")
        plt.grid(True, alpha=0.3)
        
        # SAVE WITH UNIQUE NAME
        # Replaces spaces with underscores (e.g., "Random Forest" -> "Random_Forest_results.png")
        filename = f"{model_name.replace(' ', '_')}_results.png"
        plt.savefig(filename)
        print(f"Plot saved: {filename}")
        
        # CLOSE PLOT to prevent stacking
        plt.close()