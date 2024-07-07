import lightning as L
from src.my_model.transformer import TrackFormer
from src.my_model.benchmarks import MLP
from src.utils import callbacks_list, read_time, experiment_name, stage_data
import click
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger



def stage_trainer(model, ckpts, logger, data_module, val_batches, test_batches, epochs, train_batches, **kwargs):
    """Train the models"""

    
    if model == 'TrackFormer':
        lighting_model = TrackFormer(max_iters=epochs * train_batches, **kwargs)
    else:
        lighting_model = MLP()


    # Configure Trainer
    trainer = L.Trainer(
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        limit_test_batches=test_batches,
        log_every_n_steps=100,
        max_epochs=epochs,
        logger=logger,
        callbacks=ckpts,
    )

    trainer.logger._default_hp_metric = None
    trainer.fit(lighting_model, data_module, ckpt_path="last")
    return trainer, lighting_model


@click.command()
@click.option('--model_dim', type=int, default=128, help='Model dimension')
@click.option('--num_heads', type=int, default=4, help='Number of attention heads')
@click.option('--num_layers', type=int, default=6, help='Number of transformer layers')
@click.option('--dropout', type=float, default=0.1, help='Dropout rate')
@click.option('--lr', type=float, default=5e-4, help='Learning rate')
@click.option('--warmup', type=int, default=100, help='Number of warmup steps')
@click.option('--epochs', type=int, default=100, help='Maximum number of epochs to train')
@click.option('--train_batches', type=int, default=100, help='Limit on the number of training batches per epoch')
@click.option('--exp_name', type=str, required=True, help='Name the experiment')
@click.option('--dataset', type=click.Choice(['ToyTrack', 'TrackML', "TML_RAM"]), help='Choose the dataset..')
@click.option('--loss_fn', required=True, type=click.Choice(['mse', 'qloss']), help='Choose the loss function..')
@click.option('--q', type=float, default=0.1, help='Choose the quantile function..')
@click.option('--num_workers', type=int, default=15, help='Number of workers for data loading')
@click.option('--batch_size', type=int, default=200, help='Batch size for training')
@click.option('--model', type=click.Choice(['NN', 'TrackFormer']), help='MLP or tansformer')
def main(model_dim, num_heads, num_layers, dropout, lr, warmup, epochs, train_batches, exp_name, dataset, num_workers, batch_size, loss_fn, model, q):
    """Main function"""


    # Experiment
    ckp, logger = experiment_name(exp_name)
    call = callbacks_list + ckp

    # DataModule
    data_module, num_classes, input_dim, train_batches, val_batches, test_batches = stage_data(dataset, num_workers, batch_size)

    # Train 
    trainer, _ = stage_trainer(
        model,
        call,
        logger,
        data_module,
        val_batches,
        test_batches,
        epochs=epochs,
        train_batches=train_batches,
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        quantile= q,
        warmup=warmup,
        loss_type=loss_fn
    )


    # Test on best model
    test_results = trainer.test(datamodule=data_module, ckpt_path="best", verbose=1)
    read_time()

if __name__ == '__main__':
    main()