from .context import bioms
from bioms.tools import arg, print_operator, lmatrix, expand_com, expand_anticom, compute_overlap_inds, project_vector, project

import qosy as qy
import numpy as np
import numpy.linalg as nla

# Compute a finite-difference approximation
# to a gradient, given the function.
def finite_diff_gradient(f, x, eps=1e-6):
    n = len(x)
    
    grad_f = np.zeros(n)
    for i in range(n):
        dx        = np.zeros(n)
        dx[i]     = eps
        grad_f[i] = (f(x + dx) - f(x - dx))/(2.0*eps)

    return grad_f

# Compute a finite-difference approximation
# to a hessian of a function, given the gradient.
def finite_diff_hessian(grad_f, x, eps=1e-6):
    n = len(x)
    
    hess_f  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dy          = np.zeros(n)
            dy[j]       = eps
            
            hess_f[i,j] = (grad_f(x + dy)[i] - grad_f(x - dy)[i])/(2.0*eps)

    return hess_f

# Test that the gradient and hessian formulae
# used in the minimization match the finite-difference
# results for random Hamiltonians and operators.
## NOTE: If any functions in core.py are changed,
# they need to be manually added to this test function.
def test_gradient_and_hessian():

    num_sites = 2
    basis     = qy.cluster_basis(np.arange(1,num_sites+1), np.arange(num_sites), 'Pauli')

    # The "explored" commutation and anticommutation relations
    # saved so far in the calculation. Used as a look-up table.
    explored_com_data     = [qy.Basis(), qy.Basis(), dict()]
    explored_anticom_data = [qy.Basis(), qy.Basis(), dict()]

    # The OperatorString type to use in all calculations.
    global_op_type = 'Pauli'

    # The \\lambda_1 coefficient in front of |[H, O]|^2
    coeff_com_norm = 1.0
    # The \\lambda_2 coefficient in front of |O^2 - I|^2
    coeff_binarity = 1.0

    xtol = 1e-6

    # Identity operator string for reference.
    identity = qy.opstring('I', op_type=global_op_type)

    ### DEFINE THE FUNCTIONS USED IN THE OPTIMIZATION ###
    # Variables to update at each step of the optimization.
    com_norm = None
    binarity = None
    com_residual          = None
    anticom_residual      = None
    res_anticom_ext_basis = None
    iteration_data        = None

    # Returns the iteration data
    # needed by obj, grad_obj, and hess_obj
    # and computes/updates it if it is not available.
    def updated_iteration_data(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal iteration_data, res_anticom_ext_basis
        
        if iteration_data is not None and np.allclose(y, iteration_data[0], atol=1e-14):
            return iteration_data
        else:
            [Lbar_tau, res_anticom_ext_basis] = lmatrix(basis, qy.Operator(y, basis), explored_anticom_data, operation_mode='anticommutator')
            iteration_data = [np.copy(y), Lbar_tau]

            return iteration_data

    def obj(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, L_H, com_norm, binarity, com_residual, anticom_residual, res_anticom_ext_basis

        com_residual = L_H.dot(y)
        com_norm     = nla.norm(com_residual)**2.0

        [_, Lbar_tau] = updated_iteration_data(y)

        ind_identity = res_anticom_ext_basis.index(identity)

        anticom_residual = 0.5 * Lbar_tau.dot(y)
        anticom_residual[ind_identity] -= 1.0
        binarity = nla.norm(anticom_residual)**2.0

        obj = coeff_com_norm * com_norm + coeff_binarity * binarity

        return obj

    # Specifies the gradient \partial_a Z of the objective function
    # Z = \\lambda_1 |[H, O]|^2 + \\lambda_2 |O^2 - I|^2,
    # where O=\sum_a g_a S_a.
    def grad_obj(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, C_H, res_anticom_ext_basis

        [_, Lbar_tau] = updated_iteration_data(y)

        ind_identity = res_anticom_ext_basis.index(identity)
        Lbar_vec     = Lbar_tau[ind_identity, :]
        Lbar_vec     = Lbar_vec.real

        grad_obj = coeff_com_norm * (2.0 * C_H.dot(y)) \
                   + coeff_binarity * ((Lbar_tau.H).dot(Lbar_tau.dot(y)) - 2.0 * Lbar_vec)
        grad_obj = np.array(grad_obj).flatten().real

        return grad_obj

    def hess_obj(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, C_H, explored_anticom_data

        [explored_basis, explored_extended_basis, explored_s_constants_data] = explored_anticom_data

        [_, Lbar_tau] = updated_iteration_data(y)
        Cbar_tau = (Lbar_tau.H).dot(Lbar_tau)

        # The part of the Hessian due to the commutator norm.
        hess_obj = coeff_com_norm * (2.0 * C_H)

        # The first part of the Hessian due to the binarity.
        hess_obj += coeff_binarity * (2.0 * Cbar_tau)

        # The index in the extended basis of the identity operator.
        ind_identity = res_anticom_ext_basis.index(identity)

        # Vector representation of {\\tau, \\tau} in the extended basis.
        Lbar_vec = Lbar_tau.dot(y)

        # The final terms to add to the Hessian due to the binarity.
        terms = np.zeros((len(y), len(y)), dtype=complex)

        # The remaining parts of the Hessian due to the binarity.
        for ind1 in range(len(basis)):
            os1 = basis[ind1]
            [inds_explored_eb, inds_explored_b, exp_data] = explored_s_constants_data[os1]
            for (ind_x_eb, ind_x_b, datum) in zip(inds_explored_eb, inds_explored_b, exp_data):
                os2 = explored_basis[ind_x_b]
                if os2 in basis:
                    ind2 = basis.index(os2)

                    os_eb = explored_extended_basis[ind_x_eb]
                    ind3  = res_anticom_ext_basis.index(os_eb)

                    terms[ind1, ind2] += datum * Lbar_vec[ind3]
                    if ind3 == ind_identity:
                        terms[ind1, ind2] += -2.0 * datum

        # The final terms added to the Hessian.
        hess_obj += coeff_binarity * terms

        hess_obj = np.array(hess_obj).real

        return hess_obj
    
    # Test on random Hamiltonians in the basis.
    np.random.seed(42)
    num_trials_H   = 2
    num_trials_tau = 3
    for ind_trial_H in range(num_trials_H):
        H = qy.Operator(2.0*np.random.rand(num_sites)-1.0, basis)

        ### Compute the relevant quantities in the current basis.
        [L_H, extended_basis] = lmatrix(basis, H, explored_com_data, operation_mode='commutator')
        C_H = (L_H.H).dot(L_H)
        C_H = C_H.real

        # Pick a random vector in the basis.
        for ind_trial_tau in range(num_trials_tau):
            tau = 2.0 * np.random.rand(len(basis)) - 1.0
            tau /= nla.norm(tau)

            fd_grad_tau = finite_diff_gradient(obj, tau, eps=1e-5)
            grad_tau    = grad_obj(tau)

            print('fd_grad_tau = {}'.format(fd_grad_tau))
            print('grad_tau    = {}'.format(grad_tau))
            err_grad = nla.norm(fd_grad_tau - grad_tau)
            print('err_grad = {}'.format(err_grad))
            
            assert(err_grad < 1e-8)
            
            fd_hess_tau = finite_diff_hessian(grad_obj, tau, eps=1e-4)
            hess_tau    = hess_obj(tau)

            print('fd_hess_tau = {}'.format(fd_hess_tau))
            print('hess_tau    = {}'.format(hess_tau))
            err_hess = nla.norm(fd_hess_tau - hess_tau)
            print('err_hess = {}'.format(err_hess))
            
            assert(err_hess < 1e-6)
    
