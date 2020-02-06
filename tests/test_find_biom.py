from .context import bioms
import qosy as qy
import numpy as np

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

        args = {'dbasis' : 0, 'xtol' : 1e-8, 'num_expansions' : 1, 'verbose' : True}

        [op, com_norm, binarity, results_data] = bioms.find_binary_iom(H_tb, initial_op, args)

        op *= 1.0/op.norm()
        op  = qy.convert(op, 'Majorana')
        
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
