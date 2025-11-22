import sys
import os

# Ensure the src folder is visible to Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.model_factory import ModelFactory
from src.evaluator import Evaluator

def main():
    print("Starting SuperConAI Pipeline...")
    
    # 1. Load the Data
    loader = DataLoader(r'data/train.csv')
    try:
        X_train, X_test, y_train, y_test = loader.load_and_split()
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Get all the Models used
    factory = ModelFactory()
    models = factory.get_model()
    print(f"Initialized {len(models)} models.")

    # 3. Train and Evaluate said Models
    evaluator = Evaluator()
    for name, model in models.items():
        print(f"Training {name}...")
        evaluator.evaluate(model, name, X_train, X_test, y_train, y_test)

    # 4. Summary
    evaluator.print_summary()
    
    # 5. Plot ALL Models
    print("Generating plots for all models...")
    for model_name in models.keys():
        evaluator.plot_model_results(y_test, model_name)

if __name__ == "__main__":
    main()