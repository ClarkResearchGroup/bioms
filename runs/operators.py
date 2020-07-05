import qosy as qy

def single_site_parity(site, num_orbitals, mode=None):
    """Create a single Z_i operator at the given site
    (or I-2N_i in terms of fermionic operators).

    Parameters
    ----------
    site : int
         The site i.
    num_orbitals : int
         The number of orbitals N to consider.
    mode : str, optional
         The basis B of OperatorStrings to use to
         represent the operator Z_i in. "constant"
         uses B={Z_i}. "linear" uses B={Z_j}_{j=1}^N.
         "quadratic" uses B={A_i B_j}_{i<=j=1}^N
         where A_i, B_j are Majorana fermions. 
         Defaults to "constant".

    Returns
    -------
    qosy.Operator
        The Z_i operator.

    Note
    ----
    This returns an Operator that is a sum of Majorana strings
    to accomodate different initializations. This can be converted
    to a sum of Pauli strings using qosy.convert().
    """

    if mode is None:
        mode = 'constant'
    
    # Initial operator is a D at the given site.
    coeffs     = [1.0]
    op_strings = [qy.opstring('D {}'.format(site))]
    
    if mode != 'constant':
        for site1 in range(num_orbitals):
            if mode == 'quadratic':
                for site2 in range(site1, num_orbitals):
                    if site1 != site2:
                        coeffs.append(0.0)
                        op_strings.append(qy.opstring('1j A {} B {}'.format(site1, site2)))
                        coeffs.append(0.0)
                        op_strings.append(qy.opstring('1j B {} A {}'.format(site1, site2)))
                    else:
                        if site1 != site:
                            coeffs.append(0.0)
                            op_strings.append(qy.opstring('D {}'.format(site1)))
            elif mode == 'linear':
                if site1 != site:
                    coeffs.append(0.0)
                    op_strings.append(qy.opstring('D {}'.format(site1)))
            else:
                raise ValueError('Invalid mode: {}'.format(mode))
                        

    initial_op = qy.Operator(coeffs, op_strings)

    return initial_op
