"""This examples illustrates how to setup
the graphical model from exercise 01.

We will create a gm with 3 variables 
where each variable has 2 states (0,1).
We connect all variables with a unary
factor and  all pairs of variables
with a potts factor.
"""
from dgm.models import *
from dgm.solvers import *
from dgm.value_tables import *

import dgm


variable_space = [2, 2, 2]


# create the discrete graphical model
model = dgm.models.DiscreteGraphicalModel(variable_space=variable_space)


# add unary factors:

#  phi_0(x_0) = { 0.1   if x_0  == 0
#               { 0.9   if x_0  == 1
phi_0 = dgm.value_tables.DenseValueTable(values=[0.1, 0.1])
model.add_factor(variables=[0], value_table=phi_0)

#  phi_1(x_1) = { 0.8   if x_1 == 0
#               { 0.1   if x_1 == 1
phi_1 = dgm.value_tables.DenseValueTable(values=[0.1, 0.9])
model.add_factor(variables=[1], value_table=phi_1)

#  phi_2(x_2) = { 0.1   if x_2 == 0
#               { 0.9   if x_2 == 1
phi_2 = dgm.value_tables.DenseValueTable(values=[0.9, 0.1])
model.add_factor(variables=[2], value_table=phi_2)


# add pairwise factors:
# #  phi_ij(x_i, x_j) = { 0.0   if x_i == x_j
#                       { 1.0   if x_i != x_j
beta = -1.0
phi_ij = dgm.value_tables.DenseValueTable(values=[[0,beta],
                                                  [beta,0]])

model.add_factor(variables=[0,1], value_table=phi_ij)
model.add_factor(variables=[0,2], value_table=phi_ij)
model.add_factor(variables=[1,2], value_table=phi_ij)




#run LpSolver
params = dict()
solver = LpSolver(model=model, **params)
visitor = Visitor(visit_nth=1, name='LpSolver',verbose=False)
approx_argmin = solver.optimize(visitor=visitor)
print("LpSolver   ",model.evaluate(approx_argmin))
print(approx_argmin)

