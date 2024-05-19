import lightning as L
from src.datasets.datasets import TracksDataModule
from src.my_model.transformer import RegressionTransformer
from src.utils import callbacks_list, logger
import argparse


def train_transformer(data_module, **kwargs):
    """Train the transformer model."""
    model = RegressionTransformer(max_iters=50 * 32, **kwargs)

    # Configure Trainer
    trainer = L.Trainer(
        limit_train_batches=40,
        limit_test_batches=100,
        limit_val_batches=5,
        min_steps=2,
        log_every_n_steps=1,
        max_epochs=10,
        logger=logger,
        callbacks=callbacks_list,
    )

    trainer.fit(model, data_module)
    return trainer, model


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a transformer model for track fitting', allow_abbrev=False)
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension')
    parser.add_argument('--model_dim', type=int, default=248, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup', type=int, default=100, help='Number of warmup steps')
    return parser.parse_known_args()




def main():
    """Main function."""
    args, _ = parse_arguments()  

    # DataModule
    data_module = TracksDataModule()

    # Train the model
    trainer, model = train_transformer(
        data_module,
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        warmup=args.warmup
    )

    test_results = trainer.test(model, data_module, verbose=1)
if __name__ == '__main__':
    main()