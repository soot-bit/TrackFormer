import lightning as L
from src.utils import callbacks_list, read_time, experiment_name, stage_data, stage_trainer
import click
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


@click.command()
@click.option('--model_dim', type=int, default=128, help='Model dimension')
@click.option('--num_heads', type=int, default=4, help='Number of attention heads')
@click.option('--num_layers', type=int, default=4, help='Number of transformer layers')
@click.option('--dropout', type=float, default=0.1, help='Dropout rate')
@click.option('--lr', type=float, default=5e-4, help='Learning rate')
@click.option('--warmup', type=int, default=100, help='Number of warmup steps')
@click.option('--epochs', type=int, default=2, help='Maximum number of epochs to train')
@click.option('--train_batches', type=int, default=100, help='Limit on the number of training batches per epoch')
@click.option('--exp_name', type=str, required=True, help='Name the experiment')
@click.option('--dataset',  required=True, type=click.Choice(["ToyTrack", "TrackML", "TML_RAM"]), help='Choose the dataset..')
@click.option('--loss_fn', required=True, type=str, help='Choose the loss function..[mse,mae, qloss-0.5]')
@click.option('--num_workers', type=int, default=4, help='Number of workers for data loading')
@click.option('--batch_size', type=int, default=200, help='Batch size for training')
@click.option('--model', type=click.Choice([ 'TrackFormer', 'NN']), help='MLP or tansformer')
def main(model_dim, num_heads, num_layers, dropout, lr, warmup, epochs, train_batches, exp_name, dataset, num_workers, batch_size, loss_fn, model):
    """Main function"""
    
    # Experiment
    ckp, logger = experiment_name(exp_name,loss_fn )
    call = callbacks_list + ckp

    # DataModule
    data_module, num_classes, input_dim, train_batches, val_batches, test_batches = stage_data(dataset, num_workers, batch_size)

    trainer, lighting_model = stage_trainer(
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
        warmup=warmup,
        criterion=loss_fn
    )

   
    # Test on best model
    test_results = trainer.test(datamodule=data_module, ckpt_path="best", verbose=1)
    read_time()

if __name__ == '__main__':
    main()
    read_time()
