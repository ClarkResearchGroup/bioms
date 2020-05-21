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

from bioms.tools import arg, print_operator, build_s_constants, build_l_matrix, expand_com, expand_anticom, compute_overlap_inds, project_vector, project, get_size, finite_diff_gradient, finite_diff_hessian

# Print info about memory usage.
def print_memory_usage():
    vmem = dict(psutil.virtual_memory()._asdict())
    print('vmem = {}'.format(vmem), flush=True)
    
def find_binary_iom(hamiltonian, initial_op, args=None, _check_derivatives=False, _check_quantities=False):
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
    if ('explored_basis' not in args) or (args['explored_basis'] is None):
        args['explored_basis'] = qy.Basis()
    if ('explored_com_data' not in args) or (args['explored_com_data'] is None):
        args['explored_com_data'] = dict()
    if ('explored_anticom_data' not in args) or (args['explored_anticom_data'] is None):
        args['explored_anticom_data'] = dict()
    
    # The RAM threshold.
    percent_mem_threshold = arg(args, 'percent_mem_threshold', 85.0)
    
    # If using more than the RAM threshold, empty the explored data.
    #check_memory(args)
    if verbose:
        print_memory_usage()
    
    # The OperatorString type to use in all calculations.
    # Defaults to the type of the intial_op.
    global_op_type = arg(args, 'global_op_type', initial_op.op_type)

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
    H = qy.convert(copy.deepcopy(hamiltonian), global_op_type)
    
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
    obj_val  = None
    com_residual           = None
    anticom_residual       = None
    extended_basis_anticom = None
    s_constants_anticom    = None # The current (anti-commuting) structure constants for the basis.
    iteration_data         = None

    # Returns the iteration data
    # needed by obj, grad_obj, and hess_obj
    # and computes/updates it if it is not available.
    def updated_iteration_data(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, iteration_data, extended_basis_anticom, args, s_constants_anticom
        
        if iteration_data is not None and np.allclose(y, iteration_data[0], atol=1e-14):
            return iteration_data
        else:
            Lbar_tau       = build_l_matrix(s_constants_anticom, y, basis, extended_basis_anticom)
            iteration_data = [np.copy(y), Lbar_tau]
            
            return iteration_data

    def obj(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, L_H, com_norm, binarity, obj_val, com_residual, anticom_residual, extended_basis_anticom, identity, _check_quantities
        
        com_residual = L_H.dot(y)
        com_norm     = nla.norm(com_residual)**2.0
        
        [_, Lbar_tau] = updated_iteration_data(y)
        
        ind_identity = extended_basis_anticom.index(identity)
        
        anticom_residual = 0.5 * Lbar_tau.dot(y)
        anticom_residual[ind_identity] -= 1.0
        binarity = nla.norm(anticom_residual)**2.0
        
        # For debugging, check that commutator norm and binarity are correct:
        if _check_quantities:
            optest         = qy.Operator(y, basis)
            id_op          = qy.Operator([1.], [identity])
            com_norm_check = qy.commutator(H, optest).norm() ** 2.0
            optest_sqr     = 0.5 * qy.anticommutator(optest, optest)
            binarity_check = (optest_sqr - id_op).norm() ** 2.0
            print('com_norm = {}, com_norm_check = {}'.format(com_norm, com_norm_check), flush=True)
            print('binarity = {}, binarity_check = {}'.format(binarity, binarity_check), flush=True)
            
            # Check that what the algorithm (bioms) thinks is O^2-I
            # agrees with what a brute-force calculation (qosy) thinks is O^2-I.

            # Check that they include the same operator strings.
            check_os = True
            for (_, os_) in optest_sqr:
                if os_ not in extended_basis_anticom:
                    print('extended_basis_anticom is missing {}'.format(os_), flush=True)
                    check_os = False
            for os_ in extended_basis_anticom:
                if os_ not in optest_sqr._basis and os_ != identity:
                    coeff_ = anticom_residual[extended_basis_anticom.index(os_)]
                    if np.abs(coeff_) > 1e-12:
                        print('extended_basis_anticom has an extra operator {}'.format(os_), flush=True)
                        check_os = False
            if not check_os:
                print('\nO = ', flush=True)
                print_operator(qy.Operator(y, basis), np.inf)

                print('\nWhat bioms thinks is O^2-I = ', flush=True)
                print_operator(qy.Operator(anticom_residual, extended_basis_anticom), np.inf)

                print('\nWhat qosy thinks is O^2-I =', flush=True)
                print_operator(optest_sqr - id_op, np.inf)
            assert(check_os)

            assert(np.abs(com_norm - com_norm_check) < 1e-12)
            assert(np.abs(binarity - binarity_check) < 1e-5)
        
        obj_val = coeff_com_norm * com_norm + coeff_binarity * binarity

        return obj_val

    # Specifies the gradient \partial_a Z of the objective function
    # Z = \\lambda_1 |[H, O]|^2 + \\lambda_2 |O^2 - I|^2,
    # where O=\sum_a g_a S_a.
    _checked_grad = False
    def grad_obj(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, C_H, extended_basis_anticom, identity, _check_derivatives, _checked_grad
        
        [_, Lbar_tau] = updated_iteration_data(y)
        
        ind_identity = extended_basis_anticom.index(identity)
        Lbar_vec     = Lbar_tau[ind_identity, :]
        Lbar_vec     = Lbar_vec.real

        grad_obj = coeff_com_norm * (2.0 * C_H.dot(y)) \
                   + coeff_binarity * ((Lbar_tau.H).dot(Lbar_tau.dot(y)) - 2.0 * Lbar_vec)
        grad_obj = np.array(grad_obj).flatten().real

        # For debugging, check the derivatives against finite-difference derivatives.
        if _check_derivatives and not _checked_grad:
            fd_grad_obj = finite_diff_gradient(obj, y, eps=1e-5)
            
            print('fd_grad_obj = {}'.format(fd_grad_obj), flush=True)
            print('grad_obj    = {}'.format(grad_obj), flush=True)
            err_grad = nla.norm(fd_grad_obj - grad_obj)
            print('err_grad = {}'.format(err_grad), flush=True)
            
            assert(err_grad < 1e-8)
            
            _checked_grad = True

        return grad_obj

    _checked_hess = False
    def hess_obj(y):
        # Nonlocal variables that will be used or
        # modified in this function.
        nonlocal basis, C_H, args, extended_basis_anticom, identity, _check_derivatives, _checked_hess, s_constants_anticom
        
        #explored_s_constants = args['explored_anticom_data']
        
        [_, Lbar_tau] = updated_iteration_data(y)
        Cbar_tau = (Lbar_tau.H).dot(Lbar_tau)
        
        # The part of the Hessian due to the commutator norm.
        hess_obj = coeff_com_norm * (2.0 * C_H)
        
        # The first part of the Hessian due to the binarity.
        hess_obj += coeff_binarity * (2.0 * Cbar_tau)
        
        # The index in the extended basis of the identity operator.
        ind_identity = extended_basis_anticom.index(identity)
        
        # Vector representation of {\\tau, \\tau} in the extended basis.
        Lbar_vec = Lbar_tau.dot(y)
        
        # The final terms to add to the Hessian due to the binarity.
        terms = np.zeros((len(y), len(y)), dtype=complex)
        
        # The remaining parts of the Hessian due to the binarity.
        for (indB, indC, indA, coeffC) in zip(*s_constants_anticom):
            terms[indB, indA] += coeffC * np.conj(Lbar_vec[indC])

            if indC == ind_identity:
                terms[indB, indA] += -2.0 * np.conj(coeffC)
        
        # The final terms added to the Hessian.
        hess_obj += coeff_binarity * terms

        hess_obj = np.array(hess_obj).real

        # For debugging, check the derivatives against finite-difference derivatives.
        if _check_derivatives and not _checked_hess:
            fd_hess_obj = finite_diff_hessian(grad_obj, y, eps=1e-4)
            
            print('fd_hess_obj = {}'.format(fd_hess_obj), flush=True)
            print('hess_obj    = {}'.format(hess_obj), flush=True)
            err_hess = nla.norm(fd_hess_obj - hess_obj)
            print('err_hess = {}'.format(err_hess), flush=True)
            
            assert(err_hess < 1e-6)
            
            _checked_hess = True

        return hess_obj

    com_norms  = []
    binarities = []
    taus       = []
    tau_norms  = []
    objs       = []
    def update_vars(y):
        nonlocal com_norms, binarities, taus, tau_norms, objs, iteration_data, com_norm, obj_val, binarity, _checked_grad, _checked_hess
        # Ensure that the nonlocal variables are updated properly.

        # Make sure that the vector is normalized here.
        taus.append(y/nla.norm(y))
        tau_norms.append(nla.norm(y))
        objs.append(obj_val)
        
        obj_val_original = obj_val
        
        # Called to recompute com_norm and binarity with
        # a normalized tau operator.
        obj(y/nla.norm(y))
        
        com_norms.append(com_norm)
        binarities.append(binarity)
        
        # Reset the iteration data.
        del iteration_data
        iteration_data = None

        if verbose:
            print(' (obj = {}, |\\tau| = {}, com_norm = {}, binarity = {})'.format(obj_val_original, nla.norm(y), com_norm, binarity), flush=True)

        # Reset the _checked derivatives flags so that you can check the
        # derivatives against finite difference in the next step if
        # _check_derivatives=True.
        _checked_grad = False
        _checked_hess = False

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
            
        if verbose:
            print_memory_usage()
        
        ### Compute the relevant quantities in the current basis.

        # The commutant matrix C_H is computed once in each expansion.
        [s_constants_com, extended_basis_com] = build_s_constants(basis, H._basis, args['explored_basis'], args['explored_com_data'], operation_mode='commutator')
        L_H = build_l_matrix(s_constants_com, H, basis, extended_basis_com)
        C_H = (L_H.H).dot(L_H)
        C_H = C_H.real

        # The (anti-commuting) structure constants \bar{f}_{ba}^c are computed once in each expansion.
        # But the Liouvillian matrix (L_H)_{ca} = \sum_b J_b \bar{f}_{ba}^c
        # (and the anti-commutant matrix C_H = (L_H)^\dagger L_H) are computed many times
        # from the structure constants.
        [s_constants_anticom, extended_basis_anticom] = build_s_constants(basis, basis, args['explored_basis'], args['explored_anticom_data'], operation_mode='anticommutator')
        
        basis_inds_in_explored_basis = np.array([args['explored_basis'].index(os_b) for os_b in basis], dtype=int)
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
            expand_com(H, com_residual, extended_basis_com, basis, dbasis//2, args['explored_basis'], args['explored_com_data'], truncation_size=truncation_size, verbose=verbose)
            
            # Expand by \{\tau, \tau\}
            if verbose:
                print('|Basis of \\tau^2|             = {}'.format(len(extended_basis_anticom)), flush=True)
            
            expand_anticom(anticom_residual, extended_basis_anticom, basis, dbasis//2)
            
            # Project onto the new basis.
            tau = project(basis, qy.Operator(taus[-1], old_basis))
            tau = tau.coeffs.real / nla.norm(tau.coeffs.real)

        if verbose:
            print('|Explored Basis|              = {}'.format(len(args['explored_basis'])), flush=True)
            
        if verbose and ind_expansion != num_expansions-1:
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
    initial_tau_inds   = np.array([args['explored_basis'].index(os_tv) for os_tv in initial_tau._basis], dtype=int)
    
    final_tau_vector = taus[-1]
    final_tau_inds   = basis_inds[-1]
    
    fidelities            = []
    initial_fidelities    = []
    final_fidelities      = []
    proj_final_fidelities = []
    for ind_tau in range(len(taus)):
        ind_expansion = ind_expansion_from_ind_tau[ind_tau]
        
        tau_vector = taus[ind_tau]
        tau_inds   = basis_inds[ind_expansion]
        
        # For debugging, check that the quantities stored using basis_inds
        # agree with their recorded values during the optimization.
        if _check_quantities:
            basis_op = qy.Basis([args['explored_basis'][ind_eb] for ind_eb in tau_inds])
            tau_op   = qy.Operator(tau_vector, basis_op)
            
            com_norm_recorded = com_norms[ind_tau]
            binarity_recorded = binarities[ind_tau]
            
            com_norm_check = qy.commutator(H, tau_op).norm() ** 2.0
            id_op          = qy.Operator([1.], [identity])
            binarity_check = (0.5*qy.anticommutator(tau_op, tau_op) - id_op).norm() ** 2.0
            
            print('For stored tau_{}, com_norm = {}, com_norm_check = {}'.format(ind_tau, com_norm_recorded, com_norm_check), flush=True)
            print('For stored tau_{}, binarity = {}, binarity_check = {}'.format(ind_tau, binarity_recorded, binarity_check), flush=True)
            
            assert(np.abs(com_norm_recorded - com_norm_check) < 1e-12)
            assert(np.abs(binarity_recorded - binarity_check) < 1e-9)
        
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
                    'objs'                       : objs,
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
        
        data         = [args_to_record, results_data]
        results_file = open(results_filename, 'wb')
        pickle.dump(data, results_file)
        results_file.close()

        args_to_record.clear()
        del args_to_record

    # The final optimized operator O and its commutator norm and binarity.
    tau_op   = qy.Operator(taus[-1], basis)
    com_norm = com_norms[-1]
    binarity = binarities[-1]

    # For debugging, check that the final answer's quantities agree with
    # expectations.
    if _check_quantities:
        com_norm_check = qy.commutator(H, tau_op).norm() ** 2.0
        id_op          = qy.Operator([1.], [identity])
        binarity_check = (0.5*qy.anticommutator(tau_op, tau_op) - id_op).norm() ** 2.0
        
        print('final com_norm = {}, com_norm_check = {}'.format(com_norm, com_norm_check), flush=True)
        print('final binarity = {}, binarity_check = {}'.format(binarity, binarity_check), flush=True)

        assert(np.abs(com_norm - com_norm_check) < 1e-12)
        assert(np.abs(binarity - binarity_check) < 1e-12)
        
    if verbose:
        print('Final tau = ', flush=True)
        print_operator(tau_op, num_terms=200)

        print('Final com norm  = {}'.format(com_norm), flush=True)
        print('Final binarity  = {}'.format(binarity), flush=True)
    
    return [tau_op, com_norm, binarity, results_data]
