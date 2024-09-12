import numpy as np
import logging

import sympy
from sympy import symbols, sin, cos, sinh
from sympy import Matrix
from sympy import Function
from symbolic import EinsteinTensor, MetricTensor
from symbolic.ricci import RicciTensor

#logging.getLogger('simplelog')

def setup_antidesitter():
    '''Return anti-deSitter coords & metric'''

    coords = symbols("t chi theta phi")
    t, chi, theta, phi = coords
    metric_matrix =\
            sympy.diag(-1, cos(t)**2, cos(t)**2 * sinh(chi) **2, cos(t) **2 * sinh(chi)**2 * sin(theta)**2)\
                .tolist()
    metric = MetricTensor(metric_matrix, coords)

    return coords, metric

def setup_schwarzschild():
    '''Return schwarzchild coords & metric'''

    coords = symbols("t r theta phi")
    t, r, theta, phi = coords
    metric_matrix =\
            sympy.diag(-(1-1/r), 1/(1-1/r), r**2 , r**2 * sin(theta)**2)\
                .tolist()
    metric = MetricTensor(metric_matrix, coords)

    return coords, metric

def setup_sphsym():
    '''Return schwarzchild coords & metric'''

    coords = symbols("t r theta phi")
    t, r, theta, phi = coords
    f = Function('f')
    h = Function('h')
    metric_matrix =\
            sympy.diag(-f(r), h(r), r**2 , r**2 * sin(theta)**2)\
                .tolist()
    metric = MetricTensor(metric_matrix, coords)

    return coords, metric

def setup_5D_flat():
    '''Return 5D coords & metric'''

    coords = symbols("t x y z v")
    t, x, y, z, v = coords
    At = Function('At')
    Ax = Function('Ax')
    Ay = Function('Ay')
    Az = Function('Az')
    phi = Function('phi')

    A = [At(t,x,y,z), Ax(t,x,y,z), Ay(t,x,y,z), Az(t,x,y,z), 1]

    g4 = np.diag([-1,1,1,1])
    g5 = np.diag([-1, 1, 1, 1, 0])

    metric_aslist = g5.tolist()

    for i in range(5):
        for j in range(5):
            metric_aslist[i][j] = metric_aslist[i][j] + A[i]*A[j]*phi(t,x,y,z)**2

    #print(metric_aslist)
    metric = MetricTensor(metric_aslist, coords)

    return coords, metric


def calc_curvature(coords, metric, logger):
    '''Calculate curvature tensors from metric'''

    #einst = EinsteinTensor.from_metric(metric)
    logger.info('Calculating Ricci')
    ricci = RicciTensor.from_metric(metric)
    logger.info('Finished calculating Ricci')
    curv_tensors = [ricci]
    curv_matrices = []

    for tensor in curv_tensors:
        tensor_matrix = Matrix(tensor.tensor())
        logger.info('Expanding')
        tensor_matrix = tensor_matrix.applyfunc(lambda x:sympy.expand(x))
        logger.info('Factoring')
        tensor_matrix = tensor_matrix.applyfunc(lambda x: sympy.factor(x))
        curv_matrices.append(tensor_matrix)

    return curv_matrices