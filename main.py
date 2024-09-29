import src.my_model.transformer  
import src.my_model.benchmarks
import src.datasets.datamodules
import src.utils
from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
