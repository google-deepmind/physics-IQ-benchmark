from pathlib import Path
import pandas as pd
import numpy as np
import IPython


BASEPATH = Path("/Users/carsten/projects/anates/physics-iq/results_share_neurips")

artifact_file = BASEPATH / "experiments/results/simulations/benchmark_artifact_analysis/results/artifact_pair_variance_impact.csv"



if __name__ == "__main__":
    df = pd.read_csv(artifact_file)


    IPython.embed()
