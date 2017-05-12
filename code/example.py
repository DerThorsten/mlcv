from __future__ import print_function, division, absolute_import

import numpy
import random 
import sys

from dgm.models import *
from dgm.solvers import *
from dgm.value_tables import *


numpy.random.seed(42)
random.seed(42)


if __name__ == "__main__":

    

    
    # make a grid graphical model
    n_labels = 3
    shape = [50,1]
    n_var = shape[0] * shape[1]
    variable_space = numpy.ones(n_var)*n_labels

    # construct the graphical model
    gm = DiscreteGraphicalModel(variable_space=variable_space)

    # add unaries
    

    def vi(x0,x1):
        """ @brief  convert coordiantes
            to scalar variable index
        """
        return numpy.ravel_multi_index([x0,x1], shape)



    ##################################################################
    # BUILD THE MODEL
    ##################################################################

    # add unaries
    for x0 in range(shape[0]):
        for x1 in range(shape[1]):

            values = numpy.random.rand(n_labels)
            variables = vi(x0,x1)
            gm.add_factor(variables=variables, value_table=DenseValueTable(values))

    # add second order
    potts_function = PottsFunction(shape=[n_labels, n_labels], beta=1.3)
    for x0 in range(shape[0]):
        for x1 in range(shape[1]):

            if x0 + 1 < shape[0]:
                variables = [vi(x0,x1),vi(x0+1,x1)]
                gm.add_factor(variables=variables, value_table=potts_function)
            if x1 + 1 < shape[1]:
                variables = [vi(x0,x1),vi(x0, x1+1)]
                gm.add_factor(variables=variables, value_table=potts_function)   



    ##################################################################
    # FIND ARGMIN / OPTIMIZE            
    ##################################################################
    


    #run LpSolver
    params = dict()
    icm = LpSolver(model=gm, **params)
    visitor = Visitor(visit_nth=1, name='LpSolver',verbose=False)
    approx_argmin = icm.optimize(visitor=visitor)
    print("LpSolver   ",gm.evaluate(approx_argmin))


    #run Iterated conditional modes
    params = dict(n_iterations=10000000, temp=0.02)
    icm = GibbsSampler(model=gm, **params)
    visitor = Visitor(visit_nth=10000, name='Gibbs',verbose=False, time_limit=1.0)
    approx_argmin = icm.optimize(visitor=visitor)
    print("Gibbs   ",gm.evaluate(approx_argmin))

    # run GraphCut
    params = dict(tolerance = 0.00001)
    graphcut = GraphCut(model=gm, **params)
    visitor = Visitor(name='GraphCut',verbose=False)
    approx_argmin = graphcut.optimize(visitor=visitor)
    print("GraphCut",gm.evaluate(approx_argmin))


    # run Iterated conditional modes
    params = dict()
    icm = IteratedConditionalModes(model=gm, **params)
    visitor = Visitor(visit_nth=100, name='Icm',verbose=True)
    approx_argmin = icm.optimize(visitor=visitor)
    print("Icm     ",gm.evaluate(approx_argmin))
