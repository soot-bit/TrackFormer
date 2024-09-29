import src.my_model.transformer  
import src.my_model.benchmarks
import src.datasets.datamodules
from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
