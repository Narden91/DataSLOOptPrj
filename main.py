import sys
sys.dont_write_bytecode = True

from feeder_mapping import IdealGenerator, objective_function_squared_sum, generate_all_connection_vectors
import numpy as np
  


def run_feeder_mapping(data_path: str, nc: int, nf: int, nt: int, seed: int, wires_per_feeder: int):
    # Total number of wires in the system
    nw = nf * wires_per_feeder

    print(f"Initializing IdealGenerator with {nc} consumers and {nf} feeders...")
    
    # The generator creates a ground-truth based on direct feeder assignment.
    # This is sufficient for testing the objective function.
    generator = IdealGenerator(nc=nc, nf=nf, nt=nt, data_path=data_path)
    meter_supply, line_supply, meters, topology = generator.generate(seed=seed)

    print("📊 Meter Supply Shape:", meter_supply.shape)
    print("🔌 Line Supply Shape:", line_supply.shape)
    # Note: `topology` from the generator is the ground-truth FEEDER assignment.
    print("🧩 Ground Truth Feeder Topology:", topology)
    print(f"System has {nw} wires ({wires_per_feeder} per feeder).")


    # Dummy Solution: Generate a random WIRE assignment for each consumer.
    # The decision variable `w_i` is a wire from {0, 1, ..., nw-1}.
    proposed_wire_assignments = np.random.randint(0, nw, nc)
    print(f" proposing random wire assignments: {proposed_wire_assignments[:10]}...") 

    # Call the objective function with the wire assignments.
    score = objective_function_squared_sum(
        solution=proposed_wire_assignments, 
        nc=nc, 
        nf=nf, 
        meters=meter_supply, 
        lines=line_supply,
        wires_per_feeder=wires_per_feeder
    )

    print(f"🎯 Objective Function Score (Random Wire Assignment): {score:.4f}")


if __name__ == "__main__":
    
    nc = 100
    nf = 8
    nt = 720
    wires_per_feeder = 3 # 24 wires / 8 cables = 3

    print(f"Running feeder mapping simulation with {nc} consumers, {nf} feeders, and {nt} time steps...")
    run_feeder_mapping(
        "data/clean_data.csv", 
        nc=nc, 
        nf=nf, 
        nt=nt, 
        seed=42, 
        wires_per_feeder=wires_per_feeder
    )