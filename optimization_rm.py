# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 05:06:37 2018

author: Bruno Caldas
"""
from dolfin import *
from dolfin_adjoint import *
import numpy

k = 2
inner_opt = ['l2', 'L2', 'H1']

#Declare mesh and function spaces
mesh = UnitSquareMesh(10,10)
numpy.random.seed(0)
def randomly_refine(initial_mesh, ratio_to_refine=.3):
    cf = MeshFunction('bool', initial_mesh, initial_mesh.topology().dim(), False)
    for k in range(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cf)

for i in range(k):
    mesh = randomly_refine(mesh)

V = FunctionSpace(mesh, "CG", 1) # state space
W = FunctionSpace(mesh, "DG", 0) #control space
W = FunctionSpace(mesh, "CG", 1) #control space

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
File ("out_opt/l2representation%d.pvd" %k) << dJdz #l2 representation

dJdzL2 = Function(W, name="dJdzL2")
a = TrialFunction(W)
b = TestFunction(W)
M = assemble(inner(a, b)*dx)
solve(M, dJdzL2.vector(), dJdz.vector(), annotate=False)
File("out_opt/L2representation%d.pvd" % k) << dJdzL2

dJdzH1 = Function(W, name="dJdzH1")
a = TrialFunction(W)
b = TestFunction(W)
M = assemble(inner(a, b)*dx + inner(grad(a), grad(b))*dx)
solve(M, dJdzH1.vector(), dJdz.vector(), annotate=False)
File("out_opt/H1representation%d.pvd" % k) << dJdzH1

control = Control(z)
Jhat = ReducedFunctional(J, control)

# set_log_level(ERROR)

#constraint = UFLEqualityConstraint((z-0.4)*dx, Control(z) )
#problem = MinimizationProblem(Jhat, constraints=[constraint])
problem = MinimizationProblem(Jhat)

params_dict = {
    'General': {
        'Secant':{
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage':5
        }
    },
    'Step':{
        'Type': 'Augmented Lagrangian',
        'Line Search':{
            'Descent Method':{
                'Type':'Quasi-Newtonw Method'
            },
            'Curvature Condition':{
                'Type':'Strong Wolfe Conditions'
            }
        },
        'Augmented Lagrangian':{
            'Subproblem Step Type': 'Line Search',
            'Subproblem Iteration Limit':10
        }
    },
    'Status Test':{
        'Gradient Tolerance': 1e-16,
        'Step Tolerance': 1e-15,
        'Relative Step Tolerance':1e-16,
        'Iteration Limit':1000,
    }
}

for inner_prod in inner_opt:
    solver = ROLSolver(problem, params_dict, inner_product=inner_prod)
    z.opt = solver.solve()
    File('out_opt/TOM_result_inner_%s_k_%d.pvd' % (inner_prod, k) ) << z.opt
