
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Timer, ModelSummary, Callback, RichProgressBar, RichModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from rich.table import Table
from rainbow_print import printr




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
    
    def __init__(self, monitor='val_loss', patience=6, min_delta=0.0001, verbose=True, mode='min'):
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


checkpoint_callback = ModelCheckpoint(
    dirpath="/content/aims_proj/saved_models/toytrack_Transformer",
    filename="best_model",
    save_top_k=1,
    verbose=False,
    monitor="val_loss",
    mode="min"
    )


summary = RichModelSummary()
timer = Timer()
bar = RichProgressBar()
hyper = ParmSummary()
lr_monitor = LearningRateMonitor(logging_interval="epoch")
logger = TensorBoardLogger(save_dir="/content/aims_proj/lightning_logs/", name="toytrack_Transformer")
early_stopping = OverfittingEarlyStopping(verbose=True)
callbacks_list = [checkpoint_callback, timer, hyper, lr_monitor, summary]