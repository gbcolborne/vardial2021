import torch

class CosineBertHead(torch.nn.Module):
    """Head for CosineBert, i.e. logistic regression with a single
    feature, and a positive weight w.

    """
    
    def __init__(self, enforce_w_positivity=False):
        super().__init__()
        self.enforce_w_positivity = enforce_w_positivity
        self.w = torch.nn.Parameter(torch.tensor(1).float())
        self.b = torch.nn.Parameter(torch.tensor(0).float())
        self.activation = torch.nn.Sigmoid()

        
    def forward(self, scores, return_logits=False):
        """ Forward pass.

        Args:
        - scores: batch of cosine values
        
        """
        if self.enforce_w_positivity:
            logits = torch.abs(self.w) * scores + self.b
        else:
            logits = self.w * scores + self.b
        if return_logits:
            return logits
        else:
            probs = self.activation(logits)
            return probs
        
        
