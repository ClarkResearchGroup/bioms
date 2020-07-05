import numpy as np
import qosy as qy

def xxz_chain(L, J_xy, J_z, periodic=False):
    """Construct a 1D XXZ Hamiltonian H = 1/4 \sum_<ij> [J_xy (X_i X_j + Y_i Y_j) + J_z Z_i Z_j]
    
    Parameters
    ----------
    L : int
        The length of the chain.
    J_xy : float
        The coefficient in front of the exchange term.
    J_z : float
        The coefficient in front of the Ising term.
    periodic : bool, optional
        Specifies whether the model is periodic. Defaults to False.
    
    Returns
    -------
    qosy.Operator
        The Hamiltonian.
    
    Examples
    --------
    Build a 5-site Heisenberg chain:
        >>> H = xxz_chain(5, 1.0, 1.0)
    """
    
    coeffs     = []
    op_strings = []
    for site in range(L):
        if site == L-1 and not periodic:
            continue
        sitep = (site + 1) % L

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

def xxz_square(L, J_xy, J_z, periodic=False):
    """Construct a 2D square-lattice XXZ Hamiltonian H = 1/4 \\sum_<ij> [J_xy (X_i X_j + Y_i Y_j) + J_z Z_i Z_j]
    
    Parameters
    ----------
    L : int
        The side-length of the square.
    J_xy : float
        The coefficient in front of the exchange term.
    J_z : float
        The coefficient in front of the Ising term.
    periodic : bool, optional
        Specifies whether the model is periodic. Defaults to False.
    
    Returns
    -------
    qosy.Operator
        The Hamiltonian.
    
    Examples
    --------
    Build a 5x5 2D Heisenberg model:
        >>> H = xxz_square(5, 1.0, 1.0)
    """
    
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
    """Construct a 3D cubic-lattice XXZ Hamiltonian H = 1/4 \\sum_<ij> [J_xy (X_i X_j + Y_i Y_j) + J_z Z_i Z_j]
    
    Parameters
    ----------
    L : int
        The side-length of the cubic lattice.
    J_xy : float
        The coefficient in front of the exchange term.
    J_z : float
        The coefficient in front of the Ising term.
    periodic : bool, optional
        Specifies whether the model is periodic. Defaults to False.
    
    Returns
    -------
    qosy.Operator
        The Hamiltonian.
    
    Examples
    --------
    Build a 5x5x5 3D Heisenberg model:
        >>> H = xxz_cubic(5, 1.0, 1.0)
    """
    
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

def bose_hubbard_square(L, periodic=False):
    """Construct a 2D square-lattice hard-core Bose-Hubbard Hamiltonian (without magnetic fields), 
    which when written in terms of spin operators is an XX-model of the form:
        H = -1/2 \\sum_<ij> (X_i X_j + Y_i Y_j)
    
    Parameters
    ----------
    L : int
        The side-length of the square.
    periodic : bool, optional
        Specifies whether the model is periodic. Defaults to False.
    
    Returns
    -------
    qosy.Operator
        The Hamiltonian.
    
    Examples
    --------
    Build a 5x5 2D hard-core Bose-Hubbard model:
        >>> H = xxz_square(5, 1.0, 1.0)
    """
    
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

                    for orb in ['X', 'Y']:
                        coeffs.append(-0.5) # J = 1.0
                        op_strings.append(qy.opstring('{} {} {} {}'.format(orb,s1,orb,s2)))
    
    H = qy.Operator(coeffs, op_strings)
    
    return H

def magnetic_fields(potentials):
    """Construct a magnetic fields operator H = 1/2 \\sum_j h_j Z_j from the specified potentials h_j. 
    
    Parameters
    ----------
    potentials : list or ndarray
        The potentials h_j.
    
    Returns
    -------
    qosy.Operator
        The Hamiltonian representing the magnetic fields.
    
    Examples
    --------
    Build a 5-site disordered Heisenberg model:
        >>> import numpy as np
        >>> W = 6.0 # Disorder strength
        >>> H = xxz_square(5, 1.0, 1.0) + W * magnetic_fields(2.0*np.random.rand(5) - 1.0)
    """
    
    N = len(potentials)
    
    coeffs     = 0.5 * potentials
    op_strings = [qy.opstring('Z {}'.format(site)) for site in range(N)]
    
    H = qy.Operator(coeffs, op_strings)
    
    return H

def number_ops(H, num_orbitals):
    """Construct the number operators that diagonalize the given
    quadratic fermionic tight-binding Hamiltonian.
    
    Parameters
    ----------
    H : qosy.Operator
        The quadratic tight-binding Hamiltonian.
    num_orbitals : int
        The number of orbitals (sites) in the model.
    
    Returns
    -------
    list of qosy.Operator
        The number operators that commute with H.
    """
    
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
