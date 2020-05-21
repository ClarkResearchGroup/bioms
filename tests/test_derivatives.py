from .context import bioms
from bioms import find_binary_iom

import qosy as qy
import numpy as np
import numpy.linalg as nla

# Test that the gradient and hessian formulae
# used in the minimization match the finite-difference
# results for random Hamiltonians and operators.
# No basis expansion.
def test_gradient_and_hessian1():
    num_sites = 2
    basis     = qy.cluster_basis(np.arange(1,num_sites+1), np.arange(num_sites), 'Pauli')
    
    args = {'dbasis' : 0, 'xtol' : 1e-3, 'num_expansions' : 1, 'verbose' : True, 'global_op_type' : 'Pauli'}
    
    # Pick a random Hamiltonian in the basis.
    np.random.seed(42)
    num_trials_H   = 2
    num_trials_tau = 2
    for ind_trial_H in range(num_trials_H):
        H = qy.Operator(2.0*np.random.rand(len(basis))-1.0, basis)

        # Pick a random vector in the basis.
        for ind_trial_tau in range(num_trials_tau):
            tau = 2.0 * np.random.rand(len(basis)) - 1.0
            tau /= nla.norm(tau)
            initial_op = qy.Operator(tau, basis)
            
            [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_derivatives=True)

# Test that the gradient and hessian formulae
# used in the minimization match the finite-difference
# results for random Hamiltonians and operators.
# Basis expansion also tested.
def test_gradient_and_hessian2():
    num_sites = 2
    basis     = qy.cluster_basis(np.arange(1,num_sites+1), np.arange(num_sites), 'Pauli')
    
    args = {'dbasis' : 10, 'xtol' : 1e-3, 'num_expansions' : 2, 'verbose' : True, 'global_op_type' : 'Pauli'}
    
    # Pick a random Hamiltonian in the basis.
    np.random.seed(42)
    num_trials_H   = 2
    num_trials_tau = 2
    for ind_trial_H in range(num_trials_H):
        H = qy.Operator(2.0*np.random.rand(len(basis))-1.0, basis)

        # Pick a random vector in the basis.
        for ind_trial_tau in range(num_trials_tau):
            tau = 2.0 * np.random.rand(len(basis)) - 1.0
            tau /= nla.norm(tau)
            initial_op = qy.Operator(tau, basis)
            
            [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_derivatives=True)
