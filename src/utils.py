
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Timer, ModelSummary, Callback, RichProgressBar, RichModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from rich.table import Table





class ParmSummary(Callback):
    def on_fit_start(self, trainer, pl_module):
        console = Console()

        #  table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Hyperparameter", style="bold cyan")
        table.add_column("Value", style="bold green")

        
        for name, value in pl_module.hparams.items():
            table.add_row(name.capitalize(), str(value))

        
        console.print("\n[bold magenta]Transformer Summary[/bold magenta]")
        console.print(table)
        console.print("*" * 40)




checkpoint_callback = ModelCheckpoint(
    dirpath="/content/aims_proj/saved_models/toytrack_Transformer",
    filename="best_model",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    )


summary = RichModelSummary()
timer = Timer()
bar = RichProgressBar()
hyper = ParmSummary()
lr_monitor = LearningRateMonitor(logging_interval="epoch")
logger = TensorBoardLogger(save_dir="/content/aims_proj/lightning_logs/", name="toytrack_Transformer")
early_stopping = EarlyStopping(monitor="val_loss", patience=4, min_delta=0.01, verbose=0, mode="min")

callbacks_list = [timer, hyper, lr_monitor, early_stopping, summary]