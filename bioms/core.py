import copy
import psutil
import time
import pickle
import numpy as np
import numpy.linalg as nla
import scipy.integrate as si
import scipy.optimize as so
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

import qosy as qy

from bioms.tools import arg, print_operator, lmatrix, expand_com, expand_anticom, compute_overlap_inds, project_vector, project, get_size

# Check if about to run out of memory.
# If so, empty the explored data.
def check_memory(args):
    vmem             = dict(psutil.virtual_memory()._asdict())
    percent_mem_used = float(vmem['percent'])
    if args['verbose']:
        print('vmem = {}'.format(vmem), flush=True)

        if percent_mem_used > args['percent_mem_threshold']:
            print('== EMPTYING MEMORY (percent mem used = {} > {}). =='.format(percent_mem_used, args['percent_mem_threshold']), flush=True)
            
            del args['explored_com_data']
            del args['explored_anticom_data']
            
            args['explored_com_data']     = [qy.Basis(), qy.Basis(), dict()]
            args['explored_anticom_data'] = [qy.Basis(), qy.Basis(), dict()]
            
            vmem = dict(psutil.virtual_memory()._asdict())
            print('== vmem after emptying = {} =='.format(vmem), flush=True)
    
def find_binary_iom(hamiltonian, initial_op, args=None):
    """Find an approximate binary integral of motion O by iteratively
       minimizing the objective function
          \\lambda_1 |[H, O]|^2 + \\lambda_2 |O^2 - I|^2
       using gradient descent (Newton's method with conjugate gradient
       to invert the Hessian). The optimized O is Hermitian, traceless,
       and should approximately commute with H and approximately square
       to identity.

       ...
    """
    
    if args is None:
        args = dict()
    
    ### SETUP THE ALGORITHM PARAMETERS ###

    # Flag whether to print output for the run.
    verbose = arg(args, 'verbose', False)

    # The "explored" commutation and anticommutation relations
    # saved so far in the calculation. Used as a look-up table.
    if ('explored_com_data' not in args) or (args['explored_com_data'] is None):
        args['explored_com_data'] = [qy.Basis(), qy.Basis(), dict()]
    if ('explored_anticom_data' not in args) or (args['explored_anticom_data'] is None):
        args['explored_anticom_data'] = [qy.Basis(), qy.Basis(), dict()]
    
    # The RAM threshold.
    percent_mem_threshold = arg(args, 'percent_mem_threshold', 85.0)
    
    # If using more than the RAM threshold, empty the explored data.
    check_memory(args)
    
    # The OperatorString type to use in all calculations.
    global_op_type = arg(args, 'global_op_type', 'Majorana')

    # The \\lambda_1 coefficient in front of |[H, O]|^2
    coeff_com_norm = arg(args, 'coeff_com_norm', 1.0)
    # The \\lambda_2 coefficient in front of |O^2 - I|^2
    coeff_binarity = arg(args, 'coeff_binarity', 1.0)
    
    # The size of the truncated basis to represent [H, \tau]
    # when expanding by [H, [H, \tau]].
    truncation_size = arg(args, 'truncation_size', None)
    
    # The tolerance of the answer used as a convergence criterion for Newton's method.
    xtol = arg(args, 'xtol', 1e-6)

    # The number of expansions of the basis to
    # perform during the optimization.
    num_expansions = arg(args, 'num_expansions', 6)

    # The number of OperatorStrings to add to the
    # basis at each expansion step.
    dbasis = arg(args, 'dbasis', len(initial_op._basis))

    # The filename to save data to. If not provided, do not write to file.
    results_filename = arg(args, 'results_filename', None)

    # Identity operator string for reference.
    identity = qy.opstring('I', op_type=global_op_type)
    ### SETUP THE ALGORITHM PARAMETERS ###

    ### INITIALIZE THE HAMILTONIAN AND IOM ###
    # The Hamiltonian H that we are considering.
    H = qy.convert(hamiltonian, global_op_type)
    
    # The binary integral of motion \\tau that we are finding.
    tau        = qy.convert(copy.deepcopy(initial_op), global_op_type)
    tau.coeffs = tau.coeffs.real
    tau       *= 1.0/tau.norm()

    initial_tau = copy.deepcopy(tau)

    if verbose:
        print('Initial \\tau = ', flush=True)
        print_operator(initial_tau, num_terms=20)

    basis = tau._basis
    tau   = tau.coeffs
    ### INITIALIZE THE HAMILTONIAN AND IOM ###

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
            [Lbar_tau, res_anticom_ext_basis] = lmatrix(basis, qy.Operator(y, basis), args['explored_anticom_data'], operation_mode='anticommutator')
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
        nonlocal basis, C_H, args

        [explored_basis, explored_extended_basis, explored_s_constants_data] = args['explored_anticom_data']

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

    com_norms  = []
    binarities = []
    taus       = []
    tau_norms  = []
    def update_vars(y):
        nonlocal com_norms, binarities, taus, tau_norms, iteration_data
        # Ensure that the nonlocal variables are updated properly.
        # Make sure that the vector is normalized here.
        tau_norms.append(nla.norm(y))
        obj(y/nla.norm(y))
        com_norms.append(com_norm)
        binarities.append(binarity)
        taus.append(y/nla.norm(y))

        # Reset the iteration data.
        del iteration_data
        iteration_data = None

        if verbose:
            print(' (|\\tau| = {}, com_norm = {}, binarity = {})'.format(nla.norm(y), com_norm, binarity), flush=True)

    ### DEFINE THE FUNCTIONS USED IN THE OPTIMIZATION ###

    ### RUN THE OPTIMIZATION ###
    start_run = time.time()

    prev_num_taus         = 0
    num_taus_in_expansion = []
    ind_expansion_from_ind_tau = dict()
    basis_sizes = []
    basis_inds  = []
    for ind_expansion in range(num_expansions):
        if verbose:
            print('==== Iteration {}/{} ===='.format(ind_expansion+1, num_expansions), flush=True)
            print('Basis size: {}'.format(len(basis)), flush=True)

        """
        # FOR DEBUGGING: Print the memory footprint of the different variables.
        for (var, var_name) in [(args['explored_com_data'], 'explored_com_data'), (args['explored_anticom_data'], 'explored_anticom_data'), (iteration_data, 'iteration_data'), (taus, 'taus'), (com_residual, 'com_residual'), (anticom_residual, 'anticom_residual'), (res_anticom_ext_basis, 'res_anticom_ext_basis')]:
            print('{} memory = {} MB'.format(var_name, get_size(var)/1e9))
        """
        
        ### Compute the relevant quantities in the current basis.
        [L_H, extended_basis] = lmatrix(basis, H, args['explored_com_data'], operation_mode='commutator')
        C_H = (L_H.H).dot(L_H)
        C_H = C_H.real

        explored_basis = args['explored_com_data'][0]
        basis_inds_in_explored_basis = np.array([explored_basis.index(os_b) for os_b in basis], dtype=int)
        basis_inds.append(basis_inds_in_explored_basis)

        basis_sizes.append(len(basis))

        ### Minimize the objective function.
        options    = {'maxiter' : 1000, 'disp' : verbose, 'xtol' : xtol}
        x0         = tau.real / nla.norm(tau.real)
        opt_result = so.minimize(obj, x0=x0, method='Newton-CG', jac=grad_obj, hess=hess_obj, options=options, callback=update_vars)
        
        # Clear the iteration data.
        del iteration_data
        iteration_data = None

        ### On all but the last iteration, expand the basis:
        if ind_expansion < num_expansions-1:
            old_basis = copy.deepcopy(basis)
            
            # Expand by [H, [H, \tau]]
            expand_com(H, com_residual, extended_basis, basis, dbasis//2, args['explored_com_data'], truncation_size=truncation_size, verbose=verbose)
            
            # Expand by \{\tau, \tau\}
            if verbose:
                print('|Basis of \\tau^2|             = {}'.format(len(res_anticom_ext_basis)), flush=True)
            expand_anticom(anticom_residual, res_anticom_ext_basis, basis, dbasis//2)
            
            # Project onto the new basis.
            tau = project(basis, qy.Operator(taus[-1], old_basis))
            tau = tau.coeffs.real / nla.norm(tau.coeffs.real)
        
        if verbose:
            print('\\tau = ', flush=True)
            print_operator(qy.Operator(tau, basis))
        
        ### Do some book-keeping.
        # Keep track of how many \tau's were evaluated in the current expansion.
        num_taus_in_expansion.append(len(taus) - prev_num_taus)
        for ind_tau in range(prev_num_taus, len(taus)):
            ind_expansion_from_ind_tau[ind_tau] = ind_expansion
        prev_num_taus = len(taus)

    if verbose:
        print('Computing fidelities.', flush=True)
    start = time.time()

    initial_tau_vector = initial_tau.coeffs
    explored_basis     = args['explored_com_data'][0]
    initial_tau_inds   = np.array([explored_basis.index(os_tv) for os_tv in initial_tau._basis], dtype=int)

    final_tau_vector = taus[-1]
    final_tau_inds   = basis_inds[-1]

    fidelities            = []
    initial_fidelities    = []
    final_fidelities      = []
    proj_final_fidelities = []
    for ind_tau in range(len(taus)):
        ind_expansion      = ind_expansion_from_ind_tau[ind_tau]

        tau_vector = taus[ind_tau]
        tau_inds   = basis_inds[ind_expansion]

        if ind_tau > 0:
            ind_expansion_prev = ind_expansion_from_ind_tau[ind_tau - 1] 

            prev_tau_vector = taus[ind_tau - 1]
            prev_tau_inds   = basis_inds[ind_expansion_prev]
            overlap         = compute_overlap_inds(prev_tau_vector, prev_tau_inds,
                                                   tau_vector, tau_inds)
            fidelity = np.abs(overlap)**2.0

            fidelities.append(fidelity)

            # Project the final \tau into the current tau's basis.
            proj_final_tau_vector = project_vector(final_tau_vector, final_tau_inds, tau_inds)
            # Normalize the projected vector.
            proj_final_tau_vector /= nla.norm(proj_final_tau_vector)
            # And compute its fidelity with the current tau.
            overlap_proj_final  = np.dot(np.conj(tau_vector), proj_final_tau_vector)
            fidelity_proj_final = np.abs(overlap_proj_final)**2.0

            proj_final_fidelities.append(fidelity_proj_final)

        overlap_initial  = compute_overlap_inds(initial_tau_vector, initial_tau_inds,
                                                tau_vector, tau_inds)
        fidelity_initial = np.abs(overlap_initial)**2.0
        initial_fidelities.append(fidelity_initial)

        overlap_final  = compute_overlap_inds(final_tau_vector, final_tau_inds,
                                              tau_vector, tau_inds)
        fidelity_final = np.abs(overlap_final)**2.0
        final_fidelities.append(fidelity_final)

    end = time.time()
    if verbose:
        print('Computed fidelities in {} seconds.'.format(end - start), flush=True)

    end_run = time.time()
    if verbose:
        print('Total time elapsed: {} seconds (or {} minutes or {} hours)'.format(end_run - start_run, (end_run - start_run)/60.0, (end_run - start_run)/3600.0), flush=True)

    # Store the results in a dictionary.
    results_data = {'taus'                       : taus,
                    'tau_norms'                  : tau_norms,
                    'basis_inds'                 : basis_inds,
                    'basis_sizes'                : basis_sizes,
                    'num_taus_in_expansion'      : num_taus_in_expansion,
                    'ind_expansion_from_ind_tau' : ind_expansion_from_ind_tau,
                    'com_norms'                  : com_norms,
                    'binarities'                 : binarities,
                    'fidelities'                 : fidelities,
                    'initial_fidelities'         : initial_fidelities,
                    'final_fidelities'           : final_fidelities,
                    'proj_final_fidelities'      : proj_final_fidelities}
    
    
    # Save the results to a file if provided.
    if results_filename is not None:
        # Record the input arguments in addition to the
        # results, but do not store the saved commuation
        # and anticommutation data, just the explored basis.
        args_to_record = dict()
        for key in args:
            if key not in ['explored_com_data', 'explored_anticom_data']:
                args_to_record[key] = args[key]
        args_to_record['explored_basis'] = args['explored_com_data'][0]
        
        data         = [args_to_record, results_data]
        results_file = open(results_filename, 'wb')
        pickle.dump(data, results_file)
        results_file.close()

        args_to_record.clear()
        del args_to_record

    # The final optimized operator O.
    tau_op = qy.Operator(taus[-1], basis)

    if verbose:
        print('Final tau = ', flush=True)
        print_operator(tau_op, num_terms=200)

        print('Final com norm  = {}'.format(com_norm), flush=True)
        print('Final binarity  = {}'.format(binarity), flush=True)
    
    return [tau_op, com_norms[-1], binarities[-1], results_data]
