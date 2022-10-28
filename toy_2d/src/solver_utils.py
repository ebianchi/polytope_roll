# Solver utilities

import numpy as np
import lemkelcp as lcp


EPS = 1e-5          # A small regularization term to assist the LCP solve.


"""Solve a single Linear Complementarity Problem."""
def solve_lcp(lcp_mat, lcp_vec):
    # Add in a small regularizing term to help with the LCP solve.
    lcp_mat += EPS * np.eye(lcp_mat.shape[0])

    # Solve the LCP.
    sol, exit_code, msg = lcp.lemkelcp(lcp_mat, lcp_vec, maxIter = 1000)

    assert exit_code == 0, msg
    return sol
