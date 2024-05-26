import lightning as L
from src.datasets.datasets import TracksDataModule
from src.my_model.transformer import RegressionTransformer
from src.utils import callbacks_list, logger
import argparse
import click

def train_transformer(data_module, epochs, train_batches, **kwargs):
    """Train the transformer model."""


    max_iters = epochs * train_batches
    val_batches = int(0.2 * train_batches)
    test_batches = 2 * val_batches

    model = RegressionTransformer(max_iters=max_iters, **kwargs)

    # Configure Trainer
    trainer = L.Trainer(
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        limit_test_batches=test_batches,
        log_every_n_steps=10,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks_list,
    )

    trainer.fit(model, data_module)
    return trainer, model

@click.command()
@click.option('--model_dim', type=int, default=128, help='Model dimension')
@click.option('--num_heads', type=int, default=4, help='Number of attention heads')
@click.option('--num_layers', type=int, default=6, help='Number of transformer layers')
@click.option('--dropout', type=float, default=0.1, help='Dropout rate')
@click.option('--lr', type=float, default=5e-4, help='Learning rate')
@click.option('--warmup', type=int, default=200, help='Number of warmup steps')
@click.option('--epochs', type=int, default=100, help='Maximum number of epochs to train')
@click.option('--train_batches', type=int, default=500, help='Limit on the number of training batches per epoch')
def main(model_dim, num_heads, num_layers, dropout, lr, warmup, epochs, train_batches):
    """Main function."""
    # DataModule
    data_module = TracksDataModule()

    # Train the model
    trainer, model = train_transformer(
        data_module,
        epochs=epochs,
        train_batches=train_batches,
        input_dim=2,
        model_dim=model_dim,
        num_heads=num_heads,
        num_classes=1,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        warmup=warmup
    )

    test_results = trainer.test(model, data_module, verbose=1)
    print(test_results)

if __name__ == '__main__':
    main()