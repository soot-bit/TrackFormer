import torch
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import lightning as L
import numba
from torch.nn.functional import mse_loss
from tqdm.notebook import tqdm
 

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
class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_d, output_dim):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, 512)
        self.hidden_layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(hidden_d)])
        self.output_layer = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

class ComplexNet(nn.Module):
    def __init__(self, input_size = 20, output_size = 1, hidden_sizes=(128, 64, 32)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size)
        )


    def forward(self, x):
        return self.model(x)



class MLP(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ComplexNet()


    def forward(self, x):
        x = x.flatten(-2, -1)
        x = self.model(x)
        return x

    def _calculate_loss(self, batch, mode: str):
        inp_data,_, labels,_ = batch
        preds = self(inp_data)
        loss = mse_loss(preds.squeeze(), labels[:,].squeeze())

        # Log
        self.log(f"{mode}_loss", loss, on_step=True, prog_bar=True, logger=True, batch_size=labels.numel())
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")


    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)  # L2
        return optimizer









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



import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_aggregated_attention_maps(input_data, attn_maps, idx=0, display_labels=False, concat_heads=False):
    """
    Plot aggregated attention maps from all heads for each layer.

    Parameters:
    input_data (array-like): The input data to the transformer model.
    attn_maps (list): A list of attention maps from each layer.
    idx (int): The index of the example from the batch to visualize. Default is 0.
    display_labels (bool): Whether to display labels on the axes. Default is False.
    concat_heads (bool): Whether to concatenate the attention heads or average them. Default is False (average).
    """
    # Process input data
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])

    # Extract and aggregate attention maps for each layer
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]
    if concat_heads:
        aggregated_attn_maps = [np.concatenate(m, axis=0) for m in attn_maps]  # Concatenate across heads
    else:
        aggregated_attn_maps = [np.mean(m, axis=0) for m in attn_maps]  # Average across all heads

    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4
    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*fig_size, fig_size))
    if num_layers == 1:
        ax = [ax]

    for layer in range(num_layers):
        ax[layer].imshow(aggregated_attn_maps[layer], origin='lower', vmin=0)
        if display_labels:
            ax[layer].set_xticks(list(range(seq_len)))
            ax[layer].set_xticklabels(input_data.tolist(), fontsize=6)
            ax[layer].set_yticks(list(range(seq_len)))
            ax[layer].set_yticklabels(input_data.tolist(), fontsize=6)
        ax[layer].set_title(f"Layer {layer+1}", fontsize=8)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    if concat_heads:
        plt.savefig('concatenated_attention_maps.svg', format='svg')
    else:
        plt.savefig('aggregated_attention_maps.svg', format='svg')
    plt.show()


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
