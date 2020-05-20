from .context import bioms
import qosy as qy
import numpy as np
import scipy.sparse.linalg as ssla
from bioms.tools import _explore
from bioms.hamiltonians import *

def test_explore1():
    # Tests that the _explore function works when
    # keeping the operator fixed but allowing the
    # basis to change.
    
    for mode in ['commutator', 'anticommutator']:
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()

        # Explore twice with a fixed operator
        basis1 = qy.Basis([qy.opstring(s_os) for s_os in ['Z 0', 'Z 1', 'Y 0 Y 2']])
        op     = xxz_chain(3, 1.0, 1.0, periodic=False)

        [lmatrix1, ext_basis1] = _explore(basis1, op, explored_basis, explored_s_constants_data, operation_mode=mode)

        basis2 = basis1 + qy.Basis([qy.opstring(s_os) for s_os in ['Z 2', 'X 0 X 1']])

        [lmatrix2, ext_basis2] = _explore(basis2, op, explored_basis, explored_s_constants_data, operation_mode=mode)

        # Explore once with the same operator.
        explored_basis            = qy.Basis()
        explored_extended_basis   = qy.Basis()
        explored_s_constants_data = dict()

        [lmatrix12, ext_basis12] = _explore(basis2, op, explored_basis, explored_s_constants_data, operation_mode=mode)

        # Compare results.
        diff_lmatrices = ssla.norm(lmatrix2 - lmatrix12)
        assert(diff_lmatrices < 1e-10)

        assert(ext_basis2 == ext_basis12)
    
def test_explore2():
    # Tests that the _explore function works when
    # keeping the basis fixed but allowing the
    # operator to change.

    for mode in ['commutator', 'anticommutator']:
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()

        # Explore twice with two different operators but the same basis.
        basis  = qy.Basis([qy.opstring(s_os) for s_os in ['Z 0', 'Z 1']])

        # Operator 1
        op1    = qy.Operator([0.1, 0.2, 0.3, 0.4], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2']])

        [lmatrix1, ext_basis1] = _explore(basis, op1, explored_basis, explored_s_constants_data, operation_mode=mode)

        # Operator 2 has a larger basis than operator 1
        op2 = qy.Operator([0.1, 0.2, 0.3, 0.4, -0.5, -0.6], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2', 'Z 2', 'X 2 X 3']])

        [lmatrix2, ext_basis2] = _explore(basis, op2, explored_basis, explored_s_constants_data, operation_mode=mode)

        # Explore once with the second operator and the same basis.
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()

        [lmatrix3, ext_basis3] = _explore(basis, op2, explored_basis, explored_s_constants_data, operation_mode=mode)

        print('basis      = \n{}'.format(basis))
        print('ext_basis1 = \n{}'.format(ext_basis1))
        print('ext_basis2 = \n{}'.format(ext_basis2))
        print('ext_basis3 = \n{}'.format(ext_basis3))
        
        # Compare results.
        assert(ext_basis2 == ext_basis3)
        
        diff_lmatrices = ssla.norm(lmatrix2 - lmatrix3)
        assert(diff_lmatrices < 1e-12)

def test_explore3():
    # Tests that the _explore function works when
    # allowing the basis *and* the operator to change.
    
    for mode in ['commutator', 'anticommutator']:
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()
        
        # Explore twice with two different operators and bases.
        
        # Basis 1
        basis1 = qy.Basis([qy.opstring(s_os) for s_os in ['Z 0', 'Z 1']])
        # Operator 1
        op1    = qy.Operator([0.1, 0.2, 0.3, 0.4], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2']])
        
        [lmatrix1, ext_basis1] = _explore(basis1, op1, explored_basis, explored_s_constants_data, operation_mode=mode)
        
        # Basis 2 is larger than basis 1
        basis2 = basis1 + qy.Basis([qy.opstring(s_os) for s_os in ['Z 2', 'X 0 X 1']])
        # Operator 2 has a larger basis than operator 1
        op2    = qy.Operator([0.1, 0.2, 0.3, 0.4, -0.5, -0.6], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2', 'Z 2', 'X 2 X 3']])
        
        [lmatrix2, ext_basis2] = _explore(basis2, op2, explored_basis, explored_s_constants_data, operation_mode=mode)
        
        # Explore once with the second operator and the second basis.
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()
        
        [lmatrix3, ext_basis3] = _explore(basis2, op2, explored_basis, explored_s_constants_data, operation_mode=mode)
        
        # Compare results.
        assert(ext_basis2 == ext_basis3)
        
        diff_lmatrices = ssla.norm(lmatrix2 - lmatrix3)
        assert(diff_lmatrices < 1e-12)
        
