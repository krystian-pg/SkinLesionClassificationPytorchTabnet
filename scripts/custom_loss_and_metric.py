import torch
import torch.nn.functional as F
from pytorch_tabnet.metrics import Metric

def custom_weighted_cross_entropy_loss(y_pred, y_true, class_weights):
    """
    Custom weighted cross-entropy loss function.
    """
    # Ensure that y_pred is softmaxed
    softmax_pred = torch.nn.Softmax(dim=-1)(y_pred)
    
    # Convert y_true to one-hot encoding
    y_true_one_hot = F.one_hot(y_true, num_classes=softmax_pred.size(-1)).float()
    
    # Apply the class weights
    weighted_logloss = -torch.sum(class_weights * y_true_one_hot * torch.log(softmax_pred), dim=-1)
    
    return torch.mean(weighted_logloss)


class CustomCrossEntropyMetric(Metric):
    def __init__(self, class_weights):
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self._name = "weighted_cross_entropy"
        self._maximize = False

    def __call__(self, y_true, y_score):
        y_true = torch.tensor(y_true, dtype=torch.long).to(y_score.device)
        return custom_weighted_cross_entropy_loss(y_score, y_true, self.class_weights).item()
