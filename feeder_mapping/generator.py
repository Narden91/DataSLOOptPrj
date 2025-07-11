import pandas as pd
import numpy as np

class IdealGenerator:
    """Ideal generator of data for testing purposes. It generates a grid with
        ideal meters and ideal supply. It also generates a list of meters that
        are connected to the supply."""

    def __init__(self, nc, nf, nt=720, data=None, data_path=None):
        """Initialize the ideal generator.
                Args:
                    nc (int) number of consumers in the grid
                    nf (int) number of feeders in the grid
                    nt (int) number of time steps
                    data (numpy array) data of shape (nc, nt) (optional)
                    data_path (str) path to the csv file with the data (optional)
                Returns:
                    grid: (numpy array) ideal grid - Consumers x Feeders x Time
                    real assignment: (numpy array) real assignment of consumers to feeders
                """
        self.nc = nc
        self.nf = nf
        self.nt = nt

        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            raise ValueError("Either `data` or `data_path` must be provided.")

    def _load_data(self, path):
        df = pd.read_csv(path)
        pivot = df.pivot(index="meter", columns="datetime", values="aei_value")
        return pivot.to_numpy()

    def generate(self, seed=None):
        meters = self._choose_meters(seed)
        topology = self._generate_topology(seed)
        meter_supply, line_supply = self._generate_ideal_supply(meters, topology)
        return meter_supply, line_supply, meters, topology

    def _choose_meters(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        all_consumers = self.data.shape[0]
        return np.random.choice(all_consumers, self.nc, replace=False).tolist()

    def _generate_topology(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, self.nf, self.nc)

    
    def _generate_ideal_supply(self, meters, topology):
        meters_data = self.data[meters, :self.nt]
        
        lines_data = np.zeros((self.nf, self.nt))
        for i in range(self.nf):
            lines_data[i, :] = np.sum(meters_data[topology == i, :], axis=0)
            
        return meters_data, lines_data
