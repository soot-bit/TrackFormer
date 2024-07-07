
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Timer, ModelSummary, Callback, RichProgressBar, RichModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from rich.table import Table
from rich import print
from src.datasets.datasets import ToyTrackDataModule, TrackMLDataModule, TML_RAM_DataModule
from rainbow_print import printr
import datetime




class ParmSummary(Callback):
    def on_fit_start(self, trainer, pl_module):
        console = Console()

        #  table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Hyperparameter", style="bold cyan")
        table.add_column("Value", style="bold green")

        
        for name, value in pl_module.hparams.items():
            table.add_row(name.capitalize(), str(value))

        
        console.print("\n\n**[bold magenta]🤖Transformer Summary**[/bold magenta]")
        console.print(table)
        console.print("*" * 40)


class OverfittingEarlyStopping(EarlyStopping):
    """Early stopping to prevent overfiting"""
    
    def __init__(self, monitor='val_loss', patience=15, min_delta=0.0001, verbose=True, mode='min'):
        super().__init__(monitor=monitor, patience=patience, min_delta=min_delta, verbose=verbose, mode=mode)
        self.last_loss = float('inf')
        self.increase_count = 0


    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get(self.monitor)
        if current_val_loss is None:
            raise ValueError(f"Early stopping conditioned on metric '{self.monitor}' which is not available.")
        
        if current_val_loss >= self.last_loss:
            printr("it might be overfitting")
            self.increase_count += 1
            if self.increase_count > self.patience:
                trainer.should_stop = True
                print(f"Its overfitting😱...  Early stopping...")
        else:
            self.last_loss = current_val_loss
            self.increase_count = 0
            super().on_validation_end(trainer, pl_module)

def experiment_name(exp):
    ckp = ModelCheckpoint(
                dirpath=f"/content/aims_proj/saved_models/{exp}",
                filename="best_model{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                verbose=True,
                monitor="val_loss",
                mode="min"
                )
    logger = TensorBoardLogger(save_dir="/content/aims_proj/lightning_logs/", name=f"{exp}")
    return [ckp] , logger

timer = Timer()

def read_time():
    sec = timer.time_elapsed("train")
    td = datetime.timedelta(seconds=sec)
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)

    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    # Format time 
    hmsms = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds * 1000):03}"
    print(f"Training time: {hmsms}")

def stage_data(dm, num_workers, batch_size):
    if dm == 'TrackML':
        data_module_instance = TrackMLDataModule(num_workers=num_workers, batch_size=batch_size)
        num_classes, input_dim, train_batches, val_batch, test_batch = 2, 3, data_module_instance.train_batches, None, None
    elif dm == 'ToyTrack':
        data_module_instance = ToyTrackDataModule(num_workers=num_workers, batch_size=batch_size)
        num_classes, input_dim, train_batches, val_batch, test_batch = 1, 2
    elif dm == "TML_RAM":
        data_module_instance = TML_RAM_DataModule(
            test_dir="/content/track-fitter/src/datasets/TML_datafiles/tml_hits_preprocessed_test.pt",
            train_dir="/content/track-fitter/src/datasets/TML_datafiles/tml_hits_preprocessed_train.pt",
            num_workers=num_workers,
            batch_size=batch_size
        )
        num_classes, input_dim, train_batches, val_batch, test_batch = 2, 3, data_module_instance.train_batches, None, None
    return  data_module_instance, num_classes, input_dim, train_batches, val_batch, test_batch

    
summary = RichModelSummary()
bar = RichProgressBar()
hyper = ParmSummary()
lr_monitor = LearningRateMonitor(logging_interval="epoch")
# logger = TensorBoardLogger(save_dir="/content/aims_proj/lightning_logs/", name="toytrack_Transformer")
early_stopping = OverfittingEarlyStopping(verbose=True)
callbacks_list = [ timer, hyper, lr_monitor, summary]
