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

def xxz_square(L, J_xy, J_z, periodic=False):
    Lx = L
    Ly = L
    N  = Lx*Ly
    
    coeffs     = []
    op_strings = []

    for y in range(Ly):
        for x in range(Lx):
            # Two bonds
            for bond in [(1,0), (0,1)]:
                site = y*Lx + x

                dx = bond[0]
                dy = bond[1]

                # Bond pointing to the right and up
                xp = x + dx
                yp = y + dy
                if periodic:
                    xp = xp % Lx
                    yp = yp % Ly

                if xp >= 0 and xp < Lx and yp >= 0 and yp < Ly:
                    sitep = yp*Lx + xp

                    s1 = np.minimum(site, sitep)
                    s2 = np.maximum(site, sitep)

                    for orb in ['X', 'Y', 'Z']:
                        if orb in ['X', 'Y']:
                            coeffs.append(0.25 * J_xy)
                        else:
                            coeffs.append(0.25 * J_z)
                        op_strings.append(qy.opstring('{} {} {} {}'.format(orb,s1,orb,s2)))

    H = qy.Operator(coeffs, op_strings)
    
    return H

def xxz_cubic(L, J_xy, J_z, periodic=False):
    Lx = L
    Ly = L
    Lz = L
    N  = Lx*Ly*Lz
    
    coeffs     = []
    op_strings = []

    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                # Two bonds
                for bond in [(1,0,0), (0,1,0), (0,0,1)]:
                    site = z*Lx*Ly + y*Lx + x

                    dx = bond[0]
                    dy = bond[1]
                    dz = bond[2]

                    # Bond pointing to the right, up, and in
                    xp = x + dx
                    yp = y + dy
                    zp = z + dz
                    if periodic:
                        xp = xp % Lx
                        yp = yp % Ly
                        zp = zp % Lz
                        
                    if xp >= 0 and xp < Lx and yp >= 0 and yp < Ly and zp >= 0 and zp < Lz:
                        sitep = zp*Lx*Ly + yp*Lx + xp

                        s1 = np.minimum(site, sitep)
                        s2 = np.maximum(site, sitep)

                        for orb in ['X', 'Y', 'Z']:
                            if orb in ['X', 'Y']:
                                coeffs.append(0.25 * J_xy)
                            else:
                                coeffs.append(0.25 * J_z)
                            op_strings.append(qy.opstring('{} {} {} {}'.format(orb,s1,orb,s2)))

    H = qy.Operator(coeffs, op_strings)
    
    return H

# TODO: document
def magnetic_fields(potentials):
    N = len(potentials)
    
    coeffs     = 0.5 * potentials
    op_strings = [qy.opstring('Z {}'.format(site)) for site in range(N)]
    
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
