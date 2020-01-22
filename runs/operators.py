import qosy as qy

# TODO: document
def single_site_parity(site, num_orbitals, mode=None):
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
