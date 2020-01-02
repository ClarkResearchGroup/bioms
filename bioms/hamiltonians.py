import numpy as np
import qosy as qy

# TODO: document
def xxz_chain(L, J_xy, J_z, periodic=False):
    coeffs     = []
    op_strings = []
    for site in range(L):
        if site == L-1 and not periodic:
            continue
        sitep = (site + 1) % L
        
        for orb in ['X', 'Y', 'Z']:
            if orb in ['X', 'Y']:
                coeffs.append(0.25 * J_xy)
            else:
                coeffs.append(0.25 * J_z)
            op_strings.append(qy.opstring('{} {} {} {}'.format(orb,site,orb,sitep)))
    
    H = qy.Operator(coeffs, op_strings)
    
    return H

# TODO: document
def magnetic_fields(potentials):
    L = len(potentials)
    
    coeffs     = 0.5 * potentials
    op_strings = [qy.opstring('Z {}'.format(site)) for site in range(L)]
    
    H = qy.Operator(coeffs, op_strings)
    
    return H

# TODO: 2D Hamiltonians

# TODO: document, test
def number_ops(H, num_orbitals):
    H_tb = qy.convert(H, 'Fermion')
    H_tb = H_tb.remove_zeros(tol=1e-15)
    (evals, evecs) = qy.diagonalize_quadratic_tightbinding(H_tb, num_orbitals)

    N = num_orbitals

    # NOTE: Number operator construction assumes
    # that the tight-binding Hamiltonian has only real coefficients.
    assert(np.allclose(np.imag(evecs), np.zeros_like(evecs)))
    evecs = np.real(evecs)
    
    num_ops = []
    for ind_ev in range(N):
        coeffs     = []
        op_strings = []
        for orb in range(N):
            coeffs.append(np.abs(evecs[orb, ind_ev])**2.0)
            op_strings.append(qy.opstring('D {}'.format(orb)))
        
        for orb1 in range(N):
            for orb2 in range(orb1+1,N):
                coeffs.append(-evecs[orb1, ind_ev] * evecs[orb2, ind_ev])
                op_strings.append(qy.opstring('1j A {} B {}'.format(orb1, orb2)))

                coeffs.append(+evecs[orb1, ind_ev] * evecs[orb2, ind_ev])
                op_strings.append(qy.opstring('1j B {} A {}'.format(orb1, orb2)))

        num_op = qy.Operator(coeffs, op_strings, 'Majorana')
        
        num_ops.append(num_op)

    return (num_ops, evals)
