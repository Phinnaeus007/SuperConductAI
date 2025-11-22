import unittest
import pandas as pd
import os
import sys
#to see where SuperConduct.AI is located and import from src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
#to get to the project root ie SuperConduct.AI/
project_root = os.path.dirname(current_dir)
#adds project rooot to pythons "search path"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV for testing
        self.test_csv = 'test_dummy.csv'
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'critical_temp': [5, 10, 15, 20, 25]
        })
        df.to_csv(self.test_csv, index=False)

    def test_load_and_split(self):
        # Initialize Loader
        loader = DataLoader(self.test_csv)
        
        # Run method
        X_train, X_test, y_train, y_test = loader.load_and_split()
        
        # Assertions (Checks)
        self.assertEqual(len(X_train) + len(X_test), 5) # Should have 5 rows total
        self.assertEqual(len(y_train), 4) # 80% of 5 is 4
        self.assertEqual(len(y_test), 1)  # 20% of 5 is 1

    def tearDown(self):
        # Clean up: Delete the dummy CSV
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

if __name__ == '__main__':
    unittest.main()