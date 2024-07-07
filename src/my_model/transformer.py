import lightning as L
from torch import nn, optim
from src.my_model.utils.modules import TransformerEncoder, PositionalEncoding, CosineWarmupScheduler, LossFunction
import torch
from lightning.pytorch.callbacks import  Callback


class TrackFormer(L.LightningModule):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, loss_type, quantile = 0.5, dropout=0.0, input_dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self.loss_crit =  LossFunction(loss_type, quantile)

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim)
        )

        # Positional encoding 
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)

        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)

        # regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )


    def forward(self, x, mask=None, add_positional_encoding=False):
        """
        Inputs:
            x - Input features [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs 
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        
        x = self.transformer(x, mask=mask)
        x = x.mean(dim=1) 
        x = self.regression_head(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=False):
        """
        Function for extracting the attention matrices
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    
    def _calculate_loss(self, batch, mode="train"):

        inputs, mask, label, _ = batch

        preds = self(inputs, mask, add_positional_encoding=False)
        # loss = self.loss_crit(preds.squeeze(), label.squeeze())
        loss = torch.nn.L1Loss()(preds.squeeze(), label.squeeze())
        self.log(f"{mode}_loss", loss, prog_bar=True, logger=True, batch_size=inputs.shape[0] )
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")
    
    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
    

