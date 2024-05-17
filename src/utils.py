import lightning as L
from lightning.pytorch.callbacks import  Callback




class ParmSummary(Callback):


    def on_fit_start(self, trainer, pl_module):

        data = []
        for name, value in pl_module.hparams.items():
            data.append([name.capitalize(), value])
        print()
        print("\n\n  ** Transformer Summary **")
        print(tabulate(data, headers=["Hyperparameter", "Value"], tablefmt="mixed_outline"))
        print("*"*40)
