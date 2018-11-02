import math
import numpy as np 

# import nangs
from nangs import PDE
from nangs.bocos import PeriodicBoco, DirichletBoco

# define custom PDE
class MyPDE(PDE):
    def __init__(self, inputs=None, outputs=None, params=None):
        super().__init__(inputs, outputs, params)
    def computePdeLoss(self, grads, params): 
        # here is where the magic happens
        dpdt, dpdx = grads['p']['t'], grads['p']['x']
        u = params['u']
        #print(dpdt, dpdx)
        return dpdt + u*dpdx

# define input values
x = np.linspace(0,1,10).tolist()
t = np.linspace(0,1,10).tolist()

# instanciate the PDE with inputs, outputs and parameters
pde = MyPDE(inputs={'t': t,'x': x}, outputs='p', params={'u': 1})

# periodic b.c for the space dimension
x1, x2 = 0, 1
pde.addBoco(PeriodicBoco({'x': x1, 't': t}, {'x': x2, 't': t}))

# initial condition (dirichlet for temporal dimension)
p0 = [math.sin(2.*math.pi*_x) for _x in x]
pde.addBoco(DirichletBoco({'x': x, 't': 0}, {'p': p0}))

# define solution topology
topo = {'layers': 5, 'neurons': 32, 'activations': 'relu'}
pde.buildModel(topo)

# set optimization parameters
pde.setSolverParams(lr=0.001,epochs=5000, batch_size=100)

# find the solution
pde.solve() 