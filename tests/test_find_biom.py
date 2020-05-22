from .context import bioms
import qosy as qy
import numpy as np
import numpy.linalg as nla

def test_find_biom():
    # Test that the gradient-descent works
    # by finding a non-interacting integral of
    # motion without performing any basis expansions.
    
    np.random.seed(42)
    
    # Number of random Hamiltonians to check.
    num_trials = 10
    
    # Create random non-interacting Hamiltonians.
    for ind_trial in range(num_trials):
        L    = 5
        J_xy = 1.0
        J_z  = 0.0

        # Random disorder strength.
        W = 10.0 * np.random.rand()
        
        # The XXZ chain.
        H_tb  = bioms.xxz_chain(L, J_xy, J_z)
        # Perturbing magnetic fields.
        H_tb += bioms.magnetic_fields(W * (2.0*np.random.rand(L)-1.0)) 

        # Find the fermion parity operator integral of motion
        # of the tight-binding Hamiltonian.
        [num_ops, energies] = bioms.number_ops(H_tb, L)
        #print('number operators = ')
        #qy.print_operators(num_ops)

        # Initial operator is a D at the center site.
        coeffs     = [1.0]
        op_strings = [qy.opstring('D {}'.format(L//2))]
        for site1 in range(L):
            for site2 in range(site1, L):
                if site1 != site2:
                    coeffs.append(0.0)
                    op_strings.append(qy.opstring('1j A {} B {}'.format(site1, site2)))
                    coeffs.append(0.0)
                    op_strings.append(qy.opstring('1j B {} A {}'.format(site1, site2)))
                else:
                    if site1 != L//2:
                        coeffs.append(0.0)
                        op_strings.append(qy.opstring('D {}'.format(site1)))

        initial_op = qy.Operator(coeffs, op_strings)

        args = {'dbasis' : 0, 'xtol' : 1e-8, 'num_expansions' : 1, 'verbose' : True, 'global_op_type' : 'Majorana'}

        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H_tb, initial_op, args)
        
        op *= 1.0/op.norm()
        op  = qy.convert(op, 'Majorana')

        # Check that the found operator is +/- one of the fermion parity operators.
        found_op = False
        for num_op in num_ops:
            diff1 = (num_op - op).norm()
            diff2 = (num_op + op).norm()
            print('diffs = {}, {}'.format(diff1, diff2))
            found_op = (diff1 < 1e-7) or (diff2 < 1e-7)
            if found_op:
                break

        assert(found_op)
        assert(com_norm < 1e-12)
        assert(binarity < 1e-12)
        
        # Check the commutator norm and binarity of the result.
        H = qy.convert(H_tb, 'Majorana')
        I = qy.Operator([1.], [qy.opstring('I', 'Majorana')])
        com_norm_check = qy.commutator(H, op).norm()**2.0
        binarity_check = (0.5*qy.anticommutator(op, op) - I).norm() ** 2.0

        assert(np.abs(com_norm - com_norm_check) < 1e-12)
        assert(np.abs(binarity - binarity_check) < 1e-12)
        
def test_com_norm_binarity1():
    # Check that the commutator norm and binarity
    # found by find_binary_iom() are accurate.
    # Each call to find_binary_iom is independent:
    # no explored data is reused.

    np.random.seed(43)

    # Number of random Hamiltonians to check.
    num_trials = 10

    # Identity operator.
    I = qy.Operator([1.], [qy.opstring('I', 'Pauli')])
    
    # Create random 1D Heisenberg Hamiltonians.
    for ind_trial in range(num_trials):
        L    = 5
        J_xy = 1.0
        J_z  = 1.0

        # Random disorder strength.
        W = 10.0 * np.random.rand()
        
        # The XXZ chain.
        H  = bioms.xxz_chain(L, J_xy, J_z)
        # Perturbing magnetic fields.
        H += bioms.magnetic_fields(W * (2.0*np.random.rand(L)-1.0))

        # Start with a single Z at center.
        initial_op = qy.Operator([1.], [qy.opstring('Z {}'.format(L//2))])

        print('H = \n{}'.format(H))
        print('initial_op = \n{}'.format(initial_op))

        # Allow the basis to expand.
        args = {'dbasis' : 10, 'xtol' : 1e-6, 'num_expansions' : 4, 'verbose' : True, 'global_op_type' : 'Pauli'}
        
        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_quantities=True)
        
        op *= 1.0/op.norm()

        # Check the commutator norm and binarity of the result.
        com_norm_check = qy.commutator(H, op).norm()**2.0
        binarity_check = (0.5*qy.anticommutator(op, op) - I).norm() ** 2.0

        assert(np.abs(com_norm - com_norm_check) < 1e-12)
        assert(np.abs(binarity - binarity_check) < 1e-12)

def test_com_norm_binarity2():
    # Check that the commutator norm and binarity
    # found by find_binary_iom() are accurate.
    # Each call to find_binary_iom is dependent
    # in the way that the MBL data runs are:
    # explored data is reused between runs and
    # the basis expands.

    # Allow the basis to expand.
    args = {'dbasis' : 10, 'xtol' : 1e-6, 'num_expansions' : 4, 'verbose' : True, 'global_op_type' : 'Pauli'}

    args['explored_basis']          = qy.Basis()
    args['explored_extended_basis'] = qy.Basis()
    args['explored_com_data']       = dict()
    args['explored_anticom_data']   = dict()

    np.random.seed(44)
    
    # Number of random Hamiltonians to check.
    
    num_trials = 10

    # Identity operator.
    I = qy.Operator([1.], [qy.opstring('I', 'Pauli')])
    
    # Create random 1D Heisenberg Hamiltonians.
    for ind_trial in range(num_trials):
        L    = 5
        J_xy = 1.0
        J_z  = 1.0

        # Random disorder strength.
        W = 10.0 * np.random.rand()
        
        # The XXZ chain.
        H  = bioms.xxz_chain(L, J_xy, J_z)
        # Perturbing magnetic fields.
        H += bioms.magnetic_fields(W * (2.0*np.random.rand(L)-1.0))

        # Start with a single Z at center.
        initial_op = qy.Operator([1.], [qy.opstring('Z {}'.format(L//2))])

        print('H = \n{}'.format(H))
        print('initial_op = \n{}'.format(initial_op))
        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_quantities=True)
        
        op *= 1.0/op.norm()

        # Check the commutator norm and binarity of the result.
        com_norm_check = qy.commutator(H, op).norm()**2.0
        binarity_check = (0.5*qy.anticommutator(op, op) - I).norm() ** 2.0

        assert(np.abs(com_norm - com_norm_check) < 1e-12)
        assert(np.abs(binarity - binarity_check) < 1e-12)

def test_truncation_op_type():
    # Check that the commutator norm and binarity
    # found by find_binary_iom() are accurate.
    # In this test, we check that setting a non-zero
    # truncation size for the [H, [H, tau]] truncation
    # and trying different op_types work correctly.
    
    np.random.seed(1)
    
    for global_op_type in ['Pauli', 'Majorana']:
        args = {'dbasis' : 10, 'xtol' : 1e-6, 'num_expansions' : 4, 'verbose' : True, 'global_op_type' : global_op_type, 'truncation_size' : 5}

        args['explored_basis']          = qy.Basis()
        args['explored_extended_basis'] = qy.Basis()
        args['explored_com_data']       = dict()
        args['explored_anticom_data']   = dict()

        # Number of random Hamiltonians to check.
        num_trials = 5

        # Identity operator.
        I = qy.Operator([1.], [qy.opstring('I', 'Pauli')])
        I = qy.convert(I, global_op_type)

        # Create random 1D Heisenberg Hamiltonians.
        for ind_trial in range(num_trials):
            L    = 5
            J_xy = 1.0
            J_z  = 1.0

            # Random disorder strength.
            W = 10.0 * np.random.rand()

            # The XXZ chain.
            H  = bioms.xxz_chain(L, J_xy, J_z)
            # Perturbing magnetic fields.
            H += bioms.magnetic_fields(W * (2.0*np.random.rand(L)-1.0))

            H = qy.convert(H, global_op_type)
            
            # Start with a single Z at center.
            initial_op = qy.Operator([1.], [qy.opstring('Z {}'.format(L//2))])
            initial_op = qy.convert(initial_op, global_op_type)

            print('H = \n{}'.format(H))
            print('initial_op = \n{}'.format(initial_op))
            [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_quantities=True)

            op *= 1.0/op.norm()

            # Check the commutator norm and binarity of the result.
            com_norm_check = qy.commutator(H, op).norm()**2.0
            binarity_check = (0.5*qy.anticommutator(op, op) - I).norm() ** 2.0

            assert(np.abs(com_norm - com_norm_check) < 1e-12)
            assert(np.abs(binarity - binarity_check) < 1e-12)

def test_binary_op_ED1():
    # Find a perfectly binary operator on three sites.
    # Check with ED that the resulting operator commutes
    # with the Hamiltonian and is binary.
    
    # Allow the basis to expand.
    args = {'dbasis' : 0, 'xtol' : 1e-12, 'num_expansions' : 1, 'verbose' : True, 'global_op_type' : 'Pauli'}
    
    args['explored_basis']          = qy.Basis()
    args['explored_extended_basis'] = qy.Basis()
    args['explored_com_data']       = dict()
    args['explored_anticom_data']   = dict()
    
    L     = 3
    basis = qy.cluster_basis(np.arange(1,L+1), np.arange(L), 'Pauli')

    np.random.seed(45)

    # Number of random Hamiltonians to check.
    num_trials = 2

    # Identity operator.
    I = qy.Operator([1.], [qy.opstring('I', 'Pauli')])
    
    # Create random 1D Heisenberg Hamiltonians.
    for ind_trial in range(num_trials):
        J_xy = 1.0
        J_z  = 1.0

        # Random disorder strength.
        W = 10.0 * np.random.rand()
        
        # The XXZ chain.
        H  = bioms.xxz_chain(L, J_xy, J_z)
        # Perturbing magnetic fields.
        H += bioms.magnetic_fields(W * (2.0*np.random.rand(L)-1.0))

        # Start with a single Z at center.
        Z_center             = qy.opstring('Z {}'.format(L//2))
        ind_Z_center         = basis.index(Z_center)
        coeffs               = np.zeros(len(basis), dtype=complex)
        coeffs[ind_Z_center] = 1.0
        initial_op           = qy.Operator(coeffs, basis)

        print('H = \n{}'.format(H))
        print('initial_op = \n{}'.format(initial_op))
        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_quantities=True)
        
        op *= 1.0/op.norm()

        # Check the commutator norm and binarity of the result.
        com_norm_check = qy.commutator(H, op).norm()**2.0
        binarity_check = (0.5*qy.anticommutator(op, op) - I).norm() ** 2.0

        assert(np.abs(com_norm - com_norm_check) < 1e-12)
        assert(np.abs(binarity - binarity_check) < 1e-12)

        # Check that the operator has zero commutator norm and binarity.
        assert(np.abs(com_norm) < 1e-10)
        assert(np.abs(binarity) < 1e-10)
        
        # Perform ED on the operator.
        op_mat         = qy.to_matrix(op, L).toarray()
        (evals, evecs) = nla.eigh(op_mat)
        
        print('Operator eigenvalues: {}'.format(evals))
        
        H_mat          = qy.to_matrix(H, L).toarray()
        
        binarity_ED    = np.sum(np.abs(np.abs(evals)**2.0 - 1.0)**2.0) / op_mat.shape[0]
        com_ED         = np.dot(H_mat, op_mat) - np.dot(op_mat, H_mat)
        com_norm_ED    = np.real(np.trace(np.dot(np.conj(com_ED.T), com_ED))) / com_ED.shape[0]
        
        print('com_norm       = {}'.format(com_norm))
        print('com_norm_check = {}'.format(com_norm_check))
        print('com_norm_ED    = {}'.format(com_norm_ED))
        
        print('binarity       = {}'.format(binarity))
        print('binarity_check = {}'.format(binarity_check))
        print('binarity_ED    = {}'.format(binarity_ED))

        # Check the commutator norm and binarity against the ED estimates.
        assert(np.abs(com_norm - com_norm_ED) < 1e-12)
        assert(np.abs(binarity - binarity_ED) < 1e-12)

def test_binary_op_ED2():
    # Find an approximately binary operator on five sites
    # obtained after a few basis expansions.
    # Check with ED that the commutator norm and binarity
    # agrees with the ED estimates.
    
    # Allow the basis to expand.
    args = {'dbasis' : 10, 'xtol' : 1e-12, 'num_expansions' : 2, 'verbose' : True, 'global_op_type' : 'Pauli'}
    
    args['explored_basis']          = qy.Basis()
    args['explored_extended_basis'] = qy.Basis()
    args['explored_com_data']       = dict()
    args['explored_anticom_data']   = dict()

    # Also check that the "local Hamiltonian" function in find_binary_iom()
    # is working correctly with expansions and finding an IOM efficiently. 
    L     = 11
    basis = qy.cluster_basis(np.arange(1,3+1), np.arange(L//2-1,L//2+2), 'Pauli')

    np.random.seed(46)

    # Number of random Hamiltonians to check.
    num_trials = 2

    # Identity operator.
    I = qy.Operator([1.], [qy.opstring('I', 'Pauli')])
    
    # Create random 1D Heisenberg Hamiltonians.
    for ind_trial in range(num_trials):
        J_xy = 1.0
        J_z  = 1.0

        # Random disorder strength.
        W = 10.0 * np.random.rand()
        
        # The XXZ chain.
        H  = bioms.xxz_chain(L, J_xy, J_z)
        # Perturbing magnetic fields.
        H += bioms.magnetic_fields(W * (2.0*np.random.rand(L)-1.0))

        # Start with a single Z at center.
        Z_center             = qy.opstring('Z {}'.format(L//2))
        ind_Z_center         = basis.index(Z_center)
        coeffs               = np.zeros(len(basis), dtype=complex)
        coeffs[ind_Z_center] = 1.0
        initial_op           = qy.Operator(coeffs, basis)

        print('H = \n{}'.format(H))
        print('initial_op = \n{}'.format(initial_op))
        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H, initial_op, args, _check_quantities=True)
        
        op *= 1.0/op.norm()

        # Check the commutator norm and binarity of the result.
        com_norm_check = qy.commutator(H, op).norm()**2.0
        binarity_check = (0.5*qy.anticommutator(op, op) - I).norm() ** 2.0

        assert(np.abs(com_norm - com_norm_check) < 1e-12)
        assert(np.abs(binarity - binarity_check) < 1e-12)
        
        # Perform ED on the operator.
        op_mat         = qy.to_matrix(op, L).toarray()
        (evals, evecs) = nla.eigh(op_mat)
        
        print('Operator eigenvalues: {}'.format(evals))
        
        H_mat          = qy.to_matrix(H, L).toarray()
        
        binarity_ED    = np.sum(np.abs(np.abs(evals)**2.0 - 1.0)**2.0) / op_mat.shape[0]
        com_ED         = np.dot(H_mat, op_mat) - np.dot(op_mat, H_mat)
        com_norm_ED    = np.real(np.trace(np.dot(np.conj(com_ED.T), com_ED))) / com_ED.shape[0]
        
        print('com_norm       = {}'.format(com_norm))
        print('com_norm_check = {}'.format(com_norm_check))
        print('com_norm_ED    = {}'.format(com_norm_ED))
        
        print('binarity       = {}'.format(binarity))
        print('binarity_check = {}'.format(binarity_check))
        print('binarity_ED    = {}'.format(binarity_ED))

        # Check the commutator norm and binarity against the ED estimates.
        assert(np.abs(com_norm - com_norm_ED) < 1e-12)
        assert(np.abs(binarity - binarity_ED) < 1e-12)

    
