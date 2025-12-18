import torch
from torch import nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class PQModel(nn.Module):
    def __init__(self, d):
        super(PQModel, self).__init__()
        self.P = nn.Parameter(1 * torch.eye(d, device=device) + 0.005 * torch.randn(d, d, device=device))
        self.Q = nn.Parameter(1 * torch.eye(d, device=device) + 0.005 * torch.randn(d, d, device=device))

    def forward(self, A, x):
        x = x.unsqueeze(-1)
        res = self.P @ A @ self.Q @ x
        return res.squeeze()

def train_model(Ainvn, x, y, d, lr=0.001, num_epochs=1000):
    model = PQModel(d).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(Ainvn, x)
        
        loss = nn.MSELoss()(y_pred, y)
        # loss = torch.norm(y_pred - y) / torch.norm(y)  # relative L2 error

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, MSE Loss = {loss.item()}")

    return model


def eval_model(model, Ainv, x, y, type='relative'):
    if type == 'relative':
        model.eval()
        y_pred = model(Ainv, x)
        abs_error = torch.norm(y_pred - y, p=2).item()
        rel_error = abs_error / torch.norm(y, p=2).item()
        return rel_error
    elif type == 'mse':
        model.eval()
        y_pred = model(Ainv, x)
        mse = nn.MSELoss()(y_pred, y).item()
        return mse

# Ay = x
# Ainv @ x = y
