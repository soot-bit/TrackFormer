
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