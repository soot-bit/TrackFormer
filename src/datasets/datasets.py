from typing import Optional, Union, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.utils.data import TensorDataset, Subset, random_split, DataLoader, Dataset, IterableDataset
import torch
from src.datasets.utils import ParticleGun, Detector, EventGenerator
from rich import print
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import lightning as L



            #################################
            #   Track Datasets with padding #
            #################################

class TracksDataset(IterableDataset):
    """
        Generates trackdata on the fly
    see https://github.com/ryanliu30
    """
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            lambda_: Optional[float] = 50,
            pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            warmup_t0: Optional[float] = 0,
        ):
        super().__init__()

        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.lambda_ = lambda_
        self.pt_dist = pt_dist

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        return _TrackIterable(
            self.hole_inefficiency,
            self.d0,
            self.noise,
            self.lambda_,
            self.pt_dist,
        )

class _TrackIterable:
    def __init__(
            self,
            hole_inefficiency: Optional[float] = 0,
            d0: Optional[float] = 0.1,
            noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
            lambda_: Optional[float] = 50,
            pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [1, 5],
            warmup_t0: Optional[float] = 0
        ):

        self.detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel',
            min_radius=0.5,
            max_radius=3,
            number_of_layers=10,
        )

        self.particle_gun = ParticleGun(
            dimension=2,
            num_particles=1,
            pt=pt_dist,
            pphi=[-np.pi, np.pi],
            vx=[0, d0 * 0.5**0.5, 'normal'],
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.event_gen = EventGenerator(self.particle_gun, self.detector, noise)

    def __next__(self):
        event = self.event_gen.generate_event()

        pt = event.particles.pt

        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        mask = torch.ones(x.shape[0], dtype=bool)

        return x, mask, torch.tensor([pt], dtype=torch.float), event

class TracksDatasetWrapper(Dataset):
    """ Generates and stores track data  
        ---------------------------------
    """ 
    def __init__(self, tracks_dataset: TracksDataset, num_events: int = 200):
        self.tracks_dataset = tracks_dataset()
        self.num_events = num_events
        self.events = []

        self._generate_events()

    def _generate_events(self):
        iterable = iter(self.tracks_dataset)
        for _ in range(self.num_events):
            self.events.append(next(iterable))

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        return self.events[idx]

           
           
           
            ##########################
            # Lightning Data Module  #
            ##########################
class TracksDataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset: Union[TracksDataset, TracksDatasetWrapper], 
        batch_size: int = 32, 
        num_workers: int = 4, 
        persistence: bool = True
    ):
        super().__init__()
        self.dataset = dataset()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistence = persistence


    def setup(self, stage=None):
        if isinstance(self.dataset, TracksDatasetWrapper):
            print("TracksDatasetWrapper")
            train_len = int(len(self.dataset) * 0.6)
            val_len = int(len(self.dataset) * .2)
            test_len = len(self.dataset) - train_len - val_len
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [train_len, val_len, test_len]
            )

        elif isinstance(self.dataset, TracksDataset):
            print("TracksDataset")
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset
            self.test_dataset = self.dataset
        else:
            print("Unknown dataset type:", type(self.dataset))  # just uncase 

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers= self.persistence
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
             persistent_workers= self.persistence
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers= self.persistence
        )

    @staticmethod
    def collate_fn(ls):
        """Batch maker"""
        x, mask, pt, events = zip(*ls)
        return pad_sequence(x, batch_first=True), pad_sequence(mask, batch_first=True), torch.cat(pt).squeeze(), list(events)
