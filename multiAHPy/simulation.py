import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .completion import complete_matrix
from .consistency import Consistency

class SimulationRunner:
    """
    A tool for running simulations to compare iPCM completion methods.
    """
    def __init__(self, ground_truth_matrix: np.ndarray):
        """
        Initializes the simulator with a known, complete matrix.
        """
        if np.any(ground_truth_matrix is None) or np.any(np.isnan(ground_truth_matrix)):
            raise ValueError("Ground truth matrix must be complete.")
        self.ground_truth = ground_truth_matrix.astype(float)
        self.n = self.ground_truth.shape[0]
        self.results = []

    def run_experiment(
        self,
        methods: List[str],
        num_missing_pairs_list: List[int],
        num_replications: int = 10
    ):
        """
        Runs a simulation experiment.

        For each number of missing pairs, it will:
        1. Run `num_replications` trials.
        2. In each trial, randomly remove that many pairs.
        3. Apply each completion method.
        4. Calculate and store statistics.
        """
        for num_missing in num_missing_pairs_list:
            print(f"\n--- Running simulation for {num_missing} missing pair(s) ---")
            for rep in range(num_replications):
                ipcm = self._create_ipcm(num_missing)

                for method in methods:
                    completed_matrix = complete_matrix(ipcm, method=method)
                    stats = self._calculate_stats(method, num_missing, completed_matrix)
                    self.results.append(stats)

        return pd.DataFrame(self.results)

    def _create_ipcm(self, num_missing_pairs: int) -> np.ndarray:
        """Randomly removes pairs from the ground truth matrix."""
        ipcm = self.ground_truth.copy().astype(object)

        indices = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                indices.append((i, j))

        np.random.shuffle(indices)
        indices_to_remove = indices[:num_missing_pairs]

        for i, j in indices_to_remove:
            ipcm[i, j] = None
            ipcm[j, i] = None

        return ipcm

    def _calculate_stats(self, method_name: str, num_missing: int, completed_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculates performance metrics for a single trial."""

        completed_matrix = completed_matrix.astype(float)

        mse = np.mean((self.ground_truth - completed_matrix)**2)

        from .types import Crisp
        crisp_completed_matrix = np.array([[Crisp(c) for c in row] for row in completed_matrix], dtype=object)

        gci = Consistency.calculate_gci(crisp_completed_matrix)

        cr_standard_ri = Consistency.calculate_saaty_cr(
            crisp_completed_matrix,
            num_missing_pairs=0
        )

        cr_generalized_ri = Consistency.calculate_saaty_cr(
            crisp_completed_matrix,
            num_missing_pairs=num_missing
        )

        return {
            "method": method_name,
            "num_missing": num_missing,
            "mse": mse,
            "gci": gci,
            "cr_standard": cr_standard_ri,
            "cr_generalized": cr_generalized_ri,
        }

        
