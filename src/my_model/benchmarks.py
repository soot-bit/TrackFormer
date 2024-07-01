
class CircleFit:
    def __init__(self):
        self.params = None
        self.data = None

    def _residuals(self, p, xy):
        cx, cy, r = p
        return torch.sqrt((xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2) - r

    def fit(self, xy_batch):
        self.data = xy_batch
        self.params = []
        for xy in xy_batch:
            x, y = xy[:, 0], xy[:, 1]
            init = [x.mean().item(), y.mean().item(), torch.sqrt((x - x.mean())**2 + (y - y.mean())**2).mean().item()]
            self.params.append(least_squares(self._residuals, init, args=(xy,)).x)


        return torch.tensor(self.params)[:,-1]

    def plot(self, n=5):
        if self.params is None or self.data is None:
            raise ValueError("Ohh my ..!")

        n_plots = min(n, len(self.data))
        picks = np.random.choice(range(len(self.data)), n_plots, replace=False)
        fig, ax = plt.subplots(figsize=(8, 8))

        for idx in picks:
            xy = self.data[idx]
            cx, cy, r = self.params[idx]
            ax.plot(xy[:, 0].numpy(), xy[:, 1].numpy(), 'o', label=f'Batch {idx + 1}')
            ax.add_artist(plt.Circle((cx, cy), r, fill=False, linestyle='--', label=f'Track {idx + 1} (Pt={r:.2f})'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Circle Fitting for {n_plots} Batches')
        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.grid(True)
        plt.show()



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



class ParticleRegressionModel(L.LightningModule):
    def __init__(self, NN):
        super().__init__()
        self.model = NN()


    def forward(self, x):
        x = x.flatten(-2, -1)
        x = self.model(x)
        return x

    def _calculate_loss(self, batch, mode: str):
        inp_data,_,labels,_ = batch
        preds = self(inp_data)
        loss = F.mse_loss(preds.squeeze(), labels)

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


        def train_nn(network, data_module, epochs=100, train_batches=100, val_batches=100, test_batches=200,  log_steps=40):

    lightning_model = ParticleRegressionModel(network)

    trainer = L.Trainer(
        logger=logger,
        limit_train_batches=train_batches,
        limit_test_batches=test_batches,
        limit_val_batches=val_batches,
        log_every_n_steps=log_steps,
        max_epochs=epochs,
        callbacks=[timer, checkpoint_callback, param_summary]
                        )

    trainer.logger._default_hp_metric = None



    trainer.fit(lightning_model, data_module, ckpt_path="last")
    trainer.test(datamodule=data_module, ckpt_path="best")

    read_time()

    return trainer


import torch
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, vectorize, float64
from torch.autograd import vmap

class CircleFit:
    def __init__(self):
        self.params = None
        self.data = None

    @staticmethod
    @jit(nopython=True)
    def _residuals(p, xy):
        cx, cy, r = p
        return np.sqrt((xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2) - r

    @staticmethod
    @jit(nopython=True)
    def _fit_single(xy):
        x, y = xy[:, 0], xy[:, 1]
        init = [x.mean(), y.mean(), np.sqrt((x - x.mean())**2 + (y - y.mean())**2).mean()]
        return least_squares(CircleFit._residuals, init, args=(xy,)).x

    def fit(self, xy_batch):
        self.data = xy_batch
        self.params = np.zeros((len(xy_batch), 3))
        for i in range(len(xy_batch)):
            self.params[i] = self._fit_single(xy_batch[i])
        return self.params[:, -1]

    @staticmethod
    @njit(parallel=True)
    def circle_points(radius, center, num_points):
        """
        Generates points on a circle.
        """
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        xy = np.stack((x, y), axis=1)
        return xy

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


        import torch.nn.functional as F
from sklearn.utils import resample
from scipy.stats import bootstrap


def TMLcollate_fn(batch):
    inputs, targets = zip(*batch)

    inputs = pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets, dim=0)

    return inputs, None, targets, None

loader  = DataLoader(data, batch_size=5000, shuffle=True, num_workers=10, collate_fn=TMLcollate_fn)

# models
dir_mse_TML = "/content/aims_proj/saved_models/full_TMLdataset_ram_unfiltered_last_dim/model-epoch=45-val_loss=0.226.ckpt"



# Load models
transf_mse_model = TrackingTransformer.load_from_checkpoint(dir_mse_TML)
transf_mse_model.eval().to("cpu")





# Initialize lists to store the MAE values
m_transf_mae_pt= []
m_transf_mae_pz = []

pt_conf_mae_list = []
batch_limit = 10
epochs = 1
for epoch in tqdm(range(epochs)):
    with torch.no_grad():
        for i, (input, _, target, _) in enumerate(loader):
            if i >= batch_limit:
                break
            # Transformer MSE
            pred1 = transf_mse_model(input)
            pt_mae1 = F.l1_loss(pred1[:,0], target[:,0])
            pz_mae1 = F.l1_loss(pred1[:,1], target[:,1])

            m_transf_mae_pt.append(pt_mae1.item())
            m_transf_mae_pz.append(pz_mae1.item())


            # Transformer Qloss
            pred2 = qloss_transf_tt_model(input)
            pt_mae2 = F.l1_loss(pred2[:,0], target[:,0])
            pz_mae2 = F.l1_loss(pred2[:,1], target[:,1])

            q_transf_mae_pt.append(pt_mae2.item())
            q_transf_mae_pz.append(pz_mae2.item())

            # conformal fitt squares
            pt_conf_mae = conformal_mae(input, target)
            pt_conf_mae_list.append(pt_conf_mae.item())






# Convert lists to numpy arrays
m_transf_mae_pt = np.array(m_transf_mae_pt)
# q_transf_mae_pt = np.array(q_transf_mae_pt)
m_transf_mae_pz = np.array(m_transf_mae_pz)
# q_transf_mae_pz = np.array(q_transf_mae_pz)
pt_conf_mae_list = np.array(pt_conf_mae_list)

def calculate_ci(data, confidence=0.95, n_resamples=1000):
    res = bootstrap((data,), np.mean, confidence_level=confidence, n_resamples=n_resamples, method='percentile')
    mean = np.mean(data)
    ci_lower = res.confidence_interval.low
    ci_upper = res.confidence_interval.high
    delta = (ci_upper - mean) / 2
    return mean, delta

#confidence intervals
transf_mse_pt_mean, transf_mse_pt_delta = calculate_ci(m_transf_mae_pt)
# transf_q_pt_mean, transf_q_pt_delta = calculate_ci(q_transf_mae_pt)
transf_mse_pz_mean, transf_mse_pz_delta = calculate_ci(m_transf_mae_pz)
# transf_q_pz_mean, transf_q_pz_delta = calculate_ci(q_transf_mae_pz)
pt_conf_mean, pt_conf_delta = calculate_ci(pt_conf_mae_list)


print(f"Transformer MSE PT MAE: {transf_mse_pt_mean:.4f} ± {transf_mse_pt_delta:.4f}")
# print(f"Transformer Qloss PT MAE: {transf_q_pt_mean:.4f} ± {transf_q_pt_delta:.4f}")
print(f"Transformer MSE PZ MAE: {transf_mse_pz_mean:.4f} ± {transf_mse_pz_delta:.4f}")
# print(f"Transformer Qloss PZ MAE: {transf_q_pz_mean:.4f} ± {transf_q_pz_delta:.4f}")
print(f"Conformal MAE: {pt_conf_mean:.4f} ± {pt_conf_delta:.4f}")



import numpy as np
import matplotlib.pyplot as plt

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

    from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def sk_regression(train_loader, test_loader):
    # Train
    X_train, _, y_train, _ = next(iter(train_loader))
    X_train = X_train.view(X_train.shape[0], -1)

    # Test
    X_test, _, y_test, _ = next(iter(test_loader))
    X_test = X_test.view(X_test.shape[0], -1)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    console = Console()
    table = Table(title="Linear Regression Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Train", style="green", no_wrap=True)
    table.add_column("Test", style="red", no_wrap=True)
    table.add_row("MSE", f"{train_mse:.5f}", f"{test_mse:.5f}")
    table.add_row("R-squared", f"{train_r2:.5f}", f"{test_r2:.5f}")
    console.print(table)

    plt.hist(y_train)


    #finite data with a
data_module_wrapper = TracksDataModule(False, batch_size=1000)
data_module_wrapper.setup()

train_loader = data_module_wrapper.train_dataloader()
test_loader = data_module_wrapper.test_dataloader()

sk_regression(train_loader, test_loader)