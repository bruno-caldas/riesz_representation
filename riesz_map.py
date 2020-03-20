# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 05:06:37 2018

@author: console
"""

from dolfin import *
from dolfin_adjoint import *
import numpy

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--k", type=int, default=1)
args, _=parser.parse_known_args()

#Declare mesh and function spaces
mesh = UnitSquareMesh(32,32)
numpy.random.seed(0)
def randomly_refine(initial_mesh, ratio_to_refine=.3):
    cf = MeshFunction('bool', initial_mesh, mesh.topology().dim(), False)
    for k in range(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
            pass
    return refine(initial_mesh, cf)

k = args.k
for i in range(k):
    mesh = randomly_refine(mesh)

V = FunctionSpace(mesh, "CG", 1) # state space
W = FunctionSpace(mesh, "DG", 0) #control space

z = interpolate(Constant(0), W)
z.rename("Control", "Control")
u = Function(V, name='State')
v = TestFunction(V)

F = (inner(grad(u), grad(v)) - z*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F==0, u, bc)

#Define regularisation parameter
alpha = Constant(1e-6)

#Define observation data
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1])

J = assemble((0.5*inner(u-d, u-d))*dx + 0.5*alpha *z**2*dx)

control = Control(z)
dJdz = compute_gradient(J, control)
dJdz.rename("dJdz", "dJdz")
File ("out/l2-%d.pvd" %k) << dJdz #l2 representation

dJdzL2 = Function(W, name="dJdzL2")
a = TrialFunction(W)
b = TestFunction(W)
M = assemble(inner(a, b)*dx + inner(grad(a), grad(b))*dx)
solve(M, dJdzL2.vector(), dJdz.vector(), annotate=False)

File("out/H1rep%d.pvd" % k) << dJdzL2

