#!/usr/bin/env python3
"""
Calcul effectif de la fonction de contrainte
"""

iterations = 500
backward = network_second[1]

for i in range(iterations):
   cost = constraint_func()
   print('Constraint Iteration %i, error %f' % (i, cost))

Xrecn = (np.array(backward(H).eval()) * preprocess['Xstd']) + preprocess['Xmean']
Xrecn = np.swapaxes(Xrecn, 1, 2)
Xrecn = Xrecn[0]
joints = joints_from_Xrecn(Xrecn)