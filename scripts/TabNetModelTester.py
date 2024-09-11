
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from torch.utils.data import DataLoader

# Seed
torch.manual_seed(42)
np.random.seed(42)

class TabNetModelTester:
    def __init__(self, test_dataset, num_classes=7, class_names=None, batch_size=32, device=None):
        """
        Initialize the tester with the test dataset.

        Args:
            test_dataset (torch.utils.data.Dataset): The test dataset.
            num_classes (int): Number of classes in the dataset.
            class_names (list of str): Names of the classes corresponding to the labels.
            batch_size (int): Batch size for DataLoader.
            device (torch.device, optional): Device to perform computation on. Defaults to CUDA if available.
        """
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _test_single_fold(self, model):
        """
        Test a single fold of the model.

        Args:
            model (TabNetClassifier): The trained TabNet model.

        Returns:
            dict: Classification report as a dictionary.
        """
        y_true = []
        y_pred = []

        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)  # Ensure inputs are on the correct device

            # Get predictions from the model
            preds = model.predict(inputs)  # TabNet's predict method

            if isinstance(preds[0], str):
                pred_labels = preds
            else:
                class_indices = np.argmax(preds, axis=1)  # Get the index of the max probability
                pred_labels = [self.class_names[i] for i in class_indices]  # Map indices to string labels

            y_true.extend(labels)  # Keep labels as they are, no need to convert to numpy
            y_pred.extend(pred_labels)

        return classification_report(y_true, y_pred, output_dict=True, target_names=self.class_names)

    def test_all_folds(self, best_model_paths):
        """
        Test all models across all folds.

        Args:
            best_model_paths (list): List of paths to the best model states for each fold.

        Returns:
            tuple: (reports_per_fold, mean_report)
                reports_per_fold (list): List of classification reports (dictionaries) for each fold.
                mean_report (pd.DataFrame): DataFrame containing mean results across all folds.
        """
        reports_per_fold = []

        for i, model_path in enumerate(best_model_paths):
            print(f"Testing Fold {i+1}")
            
            # Load the model
            model = TabNetClassifier()
            model.load_model(model_path)
            
            # Test the model and get the classification report
            report = self._test_single_fold(model)
            reports_per_fold.append(report)

        # Calculate mean results across all folds
        mean_report = self._compute_mean_report(reports_per_fold)

        return reports_per_fold, mean_report

    def _compute_mean_report(self, reports_per_fold):
        """
        Compute the mean classification report across all folds.

        Args:
            reports_per_fold (list): List of classification reports (dictionaries) for each fold.

        Returns:
            pd.DataFrame: DataFrame containing mean results across all folds.
        """
        report_df = pd.DataFrame()

        for report in reports_per_fold:
            fold_report_df = pd.DataFrame(report).transpose()
            report_df = report_df.add(fold_report_df, fill_value=0)

        # Averaging the reports across folds
        report_df /= len(reports_per_fold)

        return report_df
