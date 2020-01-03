import qosy as qy

# TODO: document
def single_site_parity(site, num_orbitals):
    # Initial operator is a D at the given site.
    coeffs     = [1.0]
    op_strings = [qy.opstring('D {}'.format(site))]
    for site1 in range(num_orbitals):
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

    initial_op = qy.Operator(coeffs, op_strings)

    return initial_op
