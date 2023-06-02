import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment


def check_convergence(
    losses: list,
    window_size: int=50,
    min_improvement: float=0.001,
    ) -> bool:
    """Check a loss curve for convergence.
    
    Args:
        losses: list of loss values from oldest to newest
        window_size: size of averaging windows to compare loss decrease
        min_improvement: minimum relative loss decrease between
            two windows wrt to maximum loss to rule out convergence

    """
    if len(losses) <= 4 * window_size:
        return False

    losses = np.array(losses)
    losses = losses / max(abs(losses))
    
    if abs(losses[-2*window_size:-window_size].mean() - losses[-window_size].mean()) < min_improvement:
        return True

    else:
        return False
    

def optimal_assignment(
    x1: torch.tensor,
    x2: torch.tensor,
    assignment_dim: int,
    ) -> list[np.array, np.array]:
    """Find the permutation of x2 elements along the selected dimension to
        maximize correlation with x1.

    Args:
        x1: first (fixed) tensor
        x2: second tensor
        assignment_dim: assignment dimension
    
    Returns:
        optimal assignment indices of x2 items, correlation coefficients
    """
    correlation = torch.zeros([x1.shape[assignment_dim], x2.shape[assignment_dim]])
    for i in range(x1.shape[assignment_dim]):
        for j in range(x2.shape[assignment_dim]): 
            correlation[i, j] = pearsonr(
                torch.narrow(x1, assignment_dim, i, 1).flatten(),
                torch.narrow(x2, assignment_dim, j, 1).flatten()
                )[0]
    correlation = torch.nan_to_num(correlation, 0)

    row_ind, col_ind = linear_sum_assignment(-1 * torch.abs(correlation))
    return col_ind, correlation[row_ind, col_ind].numpy()