import torch
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import lightning as L
import numba
import plotly.graph_objects as go
from torch.nn.functional import mse_loss
from tqdm.notebook import tqdm
from src.my_model.utils.modules import BaseModel

import numpy as np
import torch
from scipy.optimize import least_squares
import numba

class CircleFit:
    """
    Least squares Circle fit for ToyTrack.

    Attributes:
        params (list): List of parameters (cx, cy, r) for each fitted circle. None if not fitted yet.
        data (list): List of xy_batch data points from a torch dataloader
    """
    
    def __init__(self):
        self.params = None
        self.data = None

    @staticmethod
    @numba.jit(nopython=True)
    def _residuals(p, xy):
        cx, cy, r = p
        return np.sqrt((xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2) - r

    def fit(self, xy_batch):
        self.data = xy_batch
        self.params = []
        for xy in xy_batch:
            xy = xy.numpy()
            x, y = xy[:, 0], xy[:, 1]
            init = [x.mean(), y.mean(), np.sqrt((x - x.mean())**2 + (y - y.mean())**2).mean()]
            self.params.append(least_squares(CircleFit._residuals, init, args=(xy,)).x)

        return torch.tensor(self.params)[:, -1]

    def plot(self, n=5):
        if self.params is None or self.data is None:
            raise ValueError("Ohh my ..!")

        n_plots = min(n, len(self.data))
        picks = np.random.choice(range(len(self.data)), n_plots, replace=False)
        fig, ax = plt.subplots(figsize=(8, 8))

        for idx in picks:
            xy = self.data[idx]
            cx, cy, r = self.params[idx]
            ax.plot(xy[:, 0], xy[:, 1], 'o', label=f'Batch {idx + 1}')
            ax.add_artist(plt.Circle((cx, cy), r, fill=False, linestyle='--', label=f'Track {idx + 1} (Pt={r:.2f})'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Circle Fitting for {n_plots} Batches')
        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.grid(True)
        plt.show()



##################################################################:

                            ###################################
                            #       Neural Networks           #
                            ###################################

class ComplexNet(nn.Module):
    def __init__(self, input_size=20, output_size=1, hidden_layers=(128, 64, 32)):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(-2, -1)
        x = self.model(x)
        return x

class NeuralFit(BaseModel):
    def __init__(self, criterion, lr, max_iters, warmup, **kwargs):
        super().__init__(criterion, lr, max_iters, warmup)
        self.save_hyperparameters()
        self.model = ComplexNet(**kwargs)


    def forward(self, x):
        preds = self.model(x)
        return preds












# # Initialize lists to store the MAE values
# m_transf_mae_pt= []
# m_transf_mae_pz = []

# pt_conf_mae_list = []
# batch_limit = 10
# epochs = 1
# for epoch in tqdm(range(epochs)):
#     with torch.no_grad():
#         for i, (input, _, target, _) in enumerate(loader):
#             if i >= batch_limit:
#                 break
#             # Transformer MSE
#             pred1 = transf_mse_model(input)
#             pt_mae1 = F.l1_loss(pred1[:,0], target[:,0])
#             pz_mae1 = F.l1_loss(pred1[:,1], target[:,1])

#             m_transf_mae_pt.append(pt_mae1.item())
#             m_transf_mae_pz.append(pz_mae1.item())


#             # Transformer Qloss
#             pred2 = qloss_transf_tt_model(input)
#             pt_mae2 = F.l1_loss(pred2[:,0], target[:,0])
#             pz_mae2 = F.l1_loss(pred2[:,1], target[:,1])

#             q_transf_mae_pt.append(pt_mae2.item())
#             q_transf_mae_pz.append(pz_mae2.item())

#             # conformal fitt squares
#             pt_conf_mae = conformal_mae(input, target)
#             pt_conf_mae_list.append(pt_conf_mae.item())






# # Convert lists to numpy arrays
# m_transf_mae_pt = np.array(m_transf_mae_pt)
# # q_transf_mae_pt = np.array(q_transf_mae_pt)
# m_transf_mae_pz = np.array(m_transf_mae_pz)
# # q_transf_mae_pz = np.array(q_transf_mae_pz)
# pt_conf_mae_list = np.array(pt_conf_mae_list)

# def calculate_ci(data, confidence=0.95, n_resamples=1000):
#     res = bootstrap((data,), np.mean, confidence_level=confidence, n_resamples=n_resamples, method='percentile')
#     mean = np.mean(data)
#     ci_lower = res.confidence_interval.low
#     ci_upper = res.confidence_interval.high
#     delta = (ci_upper - mean) / 2
#     return mean, delta

# #confidence intervals
# transf_mse_pt_mean, transf_mse_pt_delta = calculate_ci(m_transf_mae_pt)
# # transf_q_pt_mean, transf_q_pt_delta = calculate_ci(q_transf_mae_pt)
# transf_mse_pz_mean, transf_mse_pz_delta = calculate_ci(m_transf_mae_pz)
# # transf_q_pz_mean, transf_q_pz_delta = calculate_ci(q_transf_mae_pz)
# pt_conf_mean, pt_conf_delta = calculate_ci(pt_conf_mae_list)


# print(f"Transformer MSE PT MAE: {transf_mse_pt_mean:.4f} ± {transf_mse_pt_delta:.4f}")
# # print(f"Transformer Qloss PT MAE: {transf_q_pt_mean:.4f} ± {transf_q_pt_delta:.4f}")
# print(f"Transformer MSE PZ MAE: {transf_mse_pz_mean:.4f} ± {transf_mse_pz_delta:.4f}")
# # print(f"Transformer Qloss PZ MAE: {transf_q_pz_mean:.4f} ± {transf_q_pz_delta:.4f}")
# print(f"Conformal MAE: {pt_conf_mean:.4f} ± {pt_conf_delta:.4f}")




def view_trajectory(inputs, mask=None):
    """
    Plots the trajectory of a particle in 3D.
    """
    if mask is not None:
        inputs = inputs[mask]

    x = inputs[:, 0].numpy()
    y = inputs[:, 1].numpy()
    z = inputs[:, 2].numpy()

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=4, color='blue', opacity=0.8),
        line=dict(color='blue', width=2)
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Particle Trajectory'
    )

    fig.show()
