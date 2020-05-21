from .context import bioms
import qosy as qy
import numpy as np
import scipy.sparse.linalg as ssla
from bioms.tools import build_s_constants, build_l_matrix
from bioms.hamiltonians import *

def test_s_constants_l_matrix1():
    # Tests that the build_s_constants and build_l_matrix functions work when
    # keeping the operator fixed but allowing the
    # basis to change.
    
    for mode in ['commutator', 'anticommutator']:
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()

        # Explore twice with a fixed operator
        basis1 = qy.Basis([qy.opstring(s_os) for s_os in ['Z 0', 'Z 1', 'Y 0 Y 2']])
        op     = xxz_chain(3, 1.0, 1.0, periodic=False)

        [s_constants1, ext_basis1] = build_s_constants(basis1, op._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix1                  = build_l_matrix(s_constants1, op, basis1, ext_basis1)

        basis2 = basis1 + qy.Basis([qy.opstring(s_os) for s_os in ['Z 2', 'X 0 X 1']])

        [s_constants2, ext_basis2] = build_s_constants(basis2, op._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix2                  = build_l_matrix(s_constants2, op, basis2, ext_basis2)

        # Explore once with the same operator.
        explored_basis            = qy.Basis()
        explored_extended_basis   = qy.Basis()
        explored_s_constants_data = dict()
        
        [s_constants12, ext_basis12] = build_s_constants(basis2, op._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix12                   = build_l_matrix(s_constants12, op, basis2, ext_basis12)
        
        # Compare results.
        diff_lmatrices = ssla.norm(l_matrix2 - l_matrix12)
        assert(diff_lmatrices < 1e-10)

        assert(ext_basis2 == ext_basis12)

def test_s_constants_l_matrix2():
    # Tests that the build_s_constants and build_l_matrix functions work when
    # keeping the basis fixed but allowing the
    # operator to change.

    for mode in ['commutator', 'anticommutator']:
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()

        # Explore twice with two different operators but the same basis.
        basis  = qy.Basis([qy.opstring(s_os) for s_os in ['Z 0', 'Z 1']])

        # Operator 1
        op1    = qy.Operator([0.1, 0.2, 0.3, 0.4], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2']])
        
        [s_constants1, ext_basis1] = build_s_constants(basis, op1._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix1                  = build_l_matrix(s_constants1, op1, basis, ext_basis1)

        # Operator 2 has a larger basis than operator 1
        op2 = qy.Operator([0.1, 0.2, 0.3, 0.4, -0.5, -0.6], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2', 'Z 2', 'X 2 X 3']])
        
        [s_constants2, ext_basis2] = build_s_constants(basis, op2._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix2                  = build_l_matrix(s_constants2, op2, basis, ext_basis2)

        # Explore once with the second operator and the same basis.
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()
        
        [s_constants3, ext_basis3] = build_s_constants(basis, op2._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix3                  = build_l_matrix(s_constants3, op2, basis, ext_basis3)

        print('basis      = \n{}'.format(basis))
        print('ext_basis1 = \n{}'.format(ext_basis1))
        print('ext_basis2 = \n{}'.format(ext_basis2))
        print('ext_basis3 = \n{}'.format(ext_basis3))
        
        # Compare results.
        assert(ext_basis2 == ext_basis3)
        
        diff_lmatrices = ssla.norm(l_matrix2 - l_matrix3)
        assert(diff_lmatrices < 1e-12)

def test_s_constants_l_matrix3():
    # Tests that the build_s_constants and build_l_matrix functions work when
    # allowing the basis *and* the operator to change.
    
    for mode in ['commutator', 'anticommutator']:
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()
        
        # Explore twice with two different operators and bases.
        
        # Basis 1
        basis1 = qy.Basis([qy.opstring(s_os) for s_os in ['Z 0', 'Z 1']])
        # Operator 1
        op1    = qy.Operator([0.1, 0.2, 0.3, 0.4], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2']])
        
        [s_constants1, ext_basis1] = build_s_constants(basis1, op1._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix1                  = build_l_matrix(s_constants1, op1, basis1, ext_basis1)
        
        # Basis 2 is larger than basis 1
        basis2 = basis1 + qy.Basis([qy.opstring(s_os) for s_os in ['Z 2', 'X 0 X 1']])
        # Operator 2 has a larger basis than operator 1
        op2    = qy.Operator([0.1, 0.2, 0.3, 0.4, -0.5, -0.6], [qy.opstring(s_os) for s_os in ['Z 0', 'X 0 X 1', 'Z 2 Z 3', 'Y 0 Y 2', 'Z 2', 'X 2 X 3']])
        
        [s_constants2, ext_basis2] = build_s_constants(basis2, op2._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix2                  = build_l_matrix(s_constants2, op2, basis2, ext_basis2)
        
        # Explore once with the second operator and the second basis.
        explored_basis            = qy.Basis()
        explored_s_constants_data = dict()
        
        [s_constants3, ext_basis3] = build_s_constants(basis2, op2._basis, explored_basis, explored_s_constants_data, operation_mode=mode)
        l_matrix3                  = build_l_matrix(s_constants3, op2, basis2, ext_basis3)
        
        # Compare results.
        assert(ext_basis2 == ext_basis3)
        
        diff_lmatrices = ssla.norm(l_matrix2 - l_matrix3)
        assert(diff_lmatrices < 1e-12)
