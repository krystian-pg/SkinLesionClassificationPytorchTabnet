import optuna
import optuna.visualization as vis
from sklearn.metrics import log_loss
from TabNetWrapper import TabNetWrapper

class HyperparameterOptimization:
    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        self.study = None

    def objective(self, trial, X_train, y_train, X_valid, y_valid):
        # Define model parameters
        n_d = trial.suggest_int('n_d', 8, 64)
        n_a = trial.suggest_int('n_a', 8, 64)
        n_steps = trial.suggest_int('n_steps', 3, 10)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        lambda_sparse = trial.suggest_float('lambda_sparse', 0.0001, 0.01)
        lr = trial.suggest_float('lr', 0.01, 0.2)

        # Initialize the TabNet model
        tabnet = TabNetWrapper(
            input_dim=X_train.shape[1],
            output_dim=len(set(y_train)),
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            lr=lr  # Pass the learning rate here
        )
        
        best_model = tabnet.train(X_train, y_train, X_valid, y_valid)
        preds = tabnet.predict(X_valid)
        loss = log_loss(y_valid, preds)
        
        return loss

    def optimize(self, X_train, y_train, X_valid, y_valid):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_valid, y_valid), n_trials=self.n_trials)
        return self.study.best_params

    # Visualization methods
    def plot_optimization_history(self):
        return vis.plot_optimization_history(self.study)

    def plot_param_importances(self):
        return vis.plot_param_importances(self.study)

    def plot_parallel_coordinate(self):
        return vis.plot_parallel_coordinate(self.study)

    def plot_slice(self):
        return vis.plot_slice(self.study)

    def plot_edf(self):
        return vis.plot_edf(self.study)

    def plot_timeline(self):
        return vis.plot_timeline(self.study)

    def plot_contour(self):
        return vis.plot_contour(self.study)

    def plot_all(self):
        # Display all relevant plots
        self.plot_optimization_history().show()
        self.plot_param_importances().show()
        self.plot_parallel_coordinate().show()
        self.plot_slice().show()
        self.plot_edf().show()
        self.plot_timeline().show()
        self.plot_contour().show()
