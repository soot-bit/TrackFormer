from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Timer, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from rich.table import Table
console = Console()




class ParmSummary(Callback):
    def on_fit_start(self, trainer, pl_module):

       
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Hyperparameter", style="bold cyan")
        table.add_column("Value", style="bold green")

        # model hyperparameters
        for name, value in pl_module.hparams.items():
            if name[0] != '_': # _class_paths
                table.add_row(name.capitalize(), str(value))
            
        table.add_row("", "") 
        #dm hyperparameters
        if trainer.datamodule is not None:
            for name, value in trainer.datamodule.hparams.items():
                if name[0] != '_': 
                    table.add_row(name.capitalize(), str(value))

        console.rule("[bold magenta]🤖TransFormer[/bold magenta]")
        console.print(table)
        console.rule("[bold magenta]Parameters[/bold magenta]")


# class OverfittingEarlyStopping(EarlyStopping):
#     """Early stopping to prevent overfiting"""
   
#     def __init__(self, monitor='val_loss', patience=15, min_delta=0.0001, verbose=True, mode='min'):
#         super().__init__(monitor=monitor, patience=patience, min_delta=min_delta, verbose=verbose, mode=mode)
#         self.last_loss = float('inf')
#         self.increase_count = 0


#     def on_validation_end(self, trainer, pl_module):
#         current_val_loss = trainer.callback_metrics.get(self.monitor)
#         if current_val_loss is None:
#             raise ValueError(f"Early stopping conditioned on metric '{self.monitor}' which is not available.")
        
#         if current_val_loss >= self.last_loss:
#             print("it might be overfitting")
#             self.increase_count += 1
#             if self.increase_count > self.patience:
#                 trainer.should_stop = True
#                 print(f"Its overfitting😱...  Early stopping...")
#         else:
#             self.last_loss = current_val_loss
#             self.increase_count = 0
#             super().on_validation_end(trainer, pl_module)



