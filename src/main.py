from src.datasets.datasets import TrackFittingDatasetFinite, TrackFittingDataset, TracksDataset
from src.my_model.transformer import RegressionTransformer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Timer, ModelSummary, Callback, RichProgressBar, RichModelSummary, TQDMProgressBar



checkpoint_callback = ModelCheckpoint(
    dirpath="/content/aims_proj/saved_models/toytrack_Transformer",
    filename="best_model",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    )

timer = Timer()
bar = RichProgressBar()
hyper = ParmSummary()
lr_monitor = LearningRateMonitor(logging_interval="epoch")
logger = TensorBoardLogger(save_dir="/content/aims_proj/lightning_logs/", name="toytrack_Transformer")
early_stopping = EarlyStopping(monitor="val_loss", patience=4, min_delta=0.5, verbose=0, mode="min")

def toytrack_train_torch(**kwargs):
    """ for training torch transformer """
    model = RegressionTransformer(max_iters=50*32, **kwargs)
    hyper = ParmSummary()


    trainer = L.Trainer(
                #limit_train_batches=40,
                #limit_test_batches=20,
                #limit_val_batches=10,
                #min_steps=10,
                log_every_n_steps=1,
                logger=logger,
                max_epochs=1000,
                callbacks=[ lr_monitor, timer]
                        )



    #data module
    # dataset = TrackFittingDataset()
    data_module = FiniteDataModule(dataset)
    trainer.fit(model, data_module)
    return trainer, model, data_module

stuff = toytrack_train_torch( input_dim=2,
                model_dim=248,
                num_heads=8,
                num_classes=1,
                num_layers=6,
                dropout=0.0,
                lr=5e-4,
                warmup=100 )