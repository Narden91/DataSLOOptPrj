from feeder_mapping.generator import IdealGenerator
from feeder_mapping.objective import objective_function
import numpy as np
import sys

sys.dont_write_bytecode = True  


def run_feeder_mapping(data_path: str, nc: int = 100, nf: int = 3, nt: int = 720, seed: int = 42):
    print(f"Initializing IdealGenerator with {nc} consumers and {nf} feeders...")

    generator = IdealGenerator(nc=nc, nf=nf, nt=nt, data_path=data_path)
    meter_supply, line_supply, meters, topology = generator.generate(seed=seed)

    print("📊 Meter Supply Shape:", meter_supply.shape)
    print("🔌 Line Supply Shape:", line_supply.shape)
    print("🧩 Ground Truth Topology:", topology)

    # Dummy: Generate a random assignment
    proposed_topology = np.random.randint(0, nf, nc)
    score = objective_function(proposed_topology, nc, nf, meters=meter_supply, lines=line_supply)

    print(f"🎯 Objective Function Score (Random): {score:.4f}")

if __name__ == "__main__":
    run_feeder_mapping("data/clean_data.csv")
