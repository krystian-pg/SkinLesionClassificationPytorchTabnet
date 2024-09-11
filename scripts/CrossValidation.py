import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTETomek
from TabNetWrapper import TabNetWrapper


class CrossValidation:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def cross_validate(self, X, y, best_params):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        all_fold_results = []
        best_models = []  # List to store the best model for each fold

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]

            # Apply SMOTETomek to the training data
            smotetomek = SMOTETomek(random_state=42)
            X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train, y_train)

            # Initialize TabNet with the best hyperparameters
            tabnet = TabNetWrapper(
                input_dim=X_train_resampled.shape[1],
                output_dim=len(set(y)),
                **best_params
            )

            # Train the model and capture the training history
            tabnet.train(X_train_resampled, y_train_resampled, X_valid, y_valid, patience=100, max_epochs=100000)
            
            # Store the best model for this fold
            best_models.append(tabnet.model)

            # Extract the history of training and validation metrics
            history = tabnet.model.history
            
            # Assuming history contains lists of metrics for each epoch
            for epoch in range(len(history['loss'])):
                all_fold_results.append({
                    'fold': fold,
                    'epoch': epoch + 1,
                    'train_loss': history['train_logloss'][epoch],
                    'train_accuracy': history['train_accuracy'][epoch],
                    'val_loss': history['valid_logloss'][epoch],
                    'val_accuracy': history['valid_accuracy'][epoch],
                })

        results_df = pd.DataFrame(all_fold_results)
        return results_df, best_models  # Return both the results and the list of best models
