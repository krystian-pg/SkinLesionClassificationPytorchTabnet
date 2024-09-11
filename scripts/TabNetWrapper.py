from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Seed
torch.manual_seed(42)


class TabNetWrapper:
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3, lr=0.02, **kwargs):
        # Initialize the optimizer with the learning rate
        optimizer_fn = torch.optim.Adam
        optimizer_params = {'lr': lr}

        # Initialize the TabNet model with the correct parameters
        self.model = TabNetClassifier(
            input_dim=input_dim, 
            output_dim=output_dim, 
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=optimizer_fn,  # Pass the optimizer function
            optimizer_params=optimizer_params,  # Pass the optimizer parameters dictionary
            **kwargs  # Pass any additional keyword arguments directly to the model
        )

    def train(self, X_train, y_train, X_valid, y_valid, patience=100, max_epochs=10000):
        # Fit the model with the specified patience and epochs
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy', 'logloss'],  # Use accuracy and logloss as metrics
            max_epochs=max_epochs,  # Specify maximum number of epochs
            patience=patience,  # Early stopping patience
            batch_size=4096, 
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )
        
        # Extract the training history
        history = {
            'train_loss': self.model.history['loss'],
            'train_accuracy': self.model.history['train_accuracy'],
            'val_loss': self.model.history['valid_logloss'],
            'val_accuracy': self.model.history['valid_accuracy']
        }

        return self.model, history

    def predict(self, X):
        # Return the predicted probabilities
        return self.model.predict_proba(X)
