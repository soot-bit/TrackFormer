from typing import Optional, Union, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.utils.data import TensorDataset, Subset, random_split, DataLoader, Dataset, IterableDataset
import torch
from src.datasets.track_utils import ParticleGun, Detector, EventGenerator



class TrackFittingDataset(IterableDataset):
    """
    The Infite IterableDataset that generates simulated particle
    ------------------------------------------------------------
   
    """
    
    def __init__(
        self,
        hole_inefficiency: Optional[float] = 0,
        d0: Optional[float] = 0.1,
        noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
        hard_proc_lambda: Optional[float] = 5,
        hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
        warmup_t0: Optional[float] = 0
    ):
        super().__init__()
        
        # Create a Detector object
        self.detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel',
            min_radius=0.5,
            max_radius=3,
            number_of_layers=10,
        )
        
        # Create a ParticleGun object
        self.hard_proc_gun = ParticleGun(
            dimension=2,
            num_particles=1, # 1
            pt=hard_proc_pt_dist,
            pphi=[-3.14, 3.14],
            vx=[0, d0 * 0.5**0.5, 'normal'],
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )
        
        # Create an EventGenerator object
        self.hard_proc_gen = EventGenerator([self.hard_proc_gun], self.detector, noise)

    
    def __iter__(self):
        while True:
            # Generate an event
            event = self.hard_proc_gen.generate_event()
            
            # Extract features and labels
            X = torch.tensor(event.hits.drop(["particle_id"], axis=1).values, dtype=torch.float)
            y = torch.tensor(event.particles[['pt']].values, dtype=torch.float).squeeze() # ['vx', 'vy', 'vz', 'pt', 'pphi', 'dimension', 'charge', 'd0', 'phi']
            

            yield X, y



class TrackFittingDatasetFinite(Dataset):
    
    """
    Simulates finite ToyTrack dataset:
    -----------------------------------
    
    Args: 
    num_events (int): Number of events to generate.
    hole_inefficiency (float, optional): Detector hole inefficiency. 
    d0 (float, optional): Parameter for particle gun distribution.
    noise (float, list, optional): Noise level or distribution parameters
    hard_proc_lambda (float, optional): Lambda parameter for hard process generation. Defaults to 5.
    hard_proc_pt_dist (float, list, optional): PT distribution parameters for hard process generation.
    warmup_t0 (float, optional): Warmup time.
    """

    def __init__(
        self,
        num_events: int = 1000,
        hole_inefficiency: Optional[float] = 0,
        d0: Optional[float] = 0.1,
        noise: Optional[Union[float, List[float], List[Union[float, str]]]] = 0,
        hard_proc_lambda: Optional[float] = 5,
        hard_proc_pt_dist: Optional[Union[float, List[float], List[Union[float, str]]]] = [100, 5, 'normal'],
        warmup_t0: Optional[float] = 0
    ):

        super().__init__()
        self.detector = Detector(
            dimension=2,
            hole_inefficiency=hole_inefficiency
        ).add_from_template(
            'barrel',
            min_radius=0.5,
            max_radius=3,
            number_of_layers=10,
        )
        self.hard_proc_gun = ParticleGun(
            dimension=2,
            num_particles=1,
            pt=hard_proc_pt_dist,
            pphi=[-3.14, 3.14],
            vx=[0, d0 * 0.5**0.5, 'normal'],
            vy=[0, d0 * 0.5**0.5, 'normal'],
        )

        self.hard_proc_gen = EventGenerator([self.hard_proc_gun], self.detector, noise)
        self.num_events = num_events
        self.X = []
        self.y = []

        # Generate data in parallel
        with Pool(processes=cpu_count()) as pool:
            for X, y in tqdm(pool.starmap(self.generate_event, [()] * self.num_events), total=self.num_events):
                self.X.append(X)
                self.y.append(y)

    def generate_event(self):
        """
        Generate a single event
        """
        event = self.hard_proc_gen.generate_event()
        X = torch.tensor(event.hits.drop(["particle_id"], axis=1).values, dtype=torch.float)
        y = torch.tensor(event.particles[['pt']].values, dtype=torch.float).squeeze()
        return X, y

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]