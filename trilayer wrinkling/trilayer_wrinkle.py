'''

This code uses PETSs TAO solver for nonlinear (bound-constrained)
optimization problems to solve a mechanical instatbility problem
in FEniCS. The code is largely inspired by the demo code for buckle
problem in FEniCS project. 

We consider here a trilayer model consisting of biofilm (f), debris (d)
and agar substrate(s). All three layers are modeled as neo-Hookean 
materials. The system is constrained in a box, which mimics the physical 
constraint on biofilms when grown on agar susbtrate. The growth-induced 
compression drives the biofilm surface undulations. 

In contrast with the bilayer model, the biofilm can undergo wrinkling 
instability even when the substrate is stiffer, due to the presence of 
the intermediate soft debris layer which soften the substrate surface 
effectively (Lejeune et al. Soft Matter 2016).

- Chenyi Fei (cfei@princeton.edu)
- Sheng Mao (mao.sheng07@gmail.com)

Updated: 2018-05-17

'''

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

##################################################
### 1. Parameter Setting

Min_Epsilon = 0.;
Max_Epsilon = 0.6;
# NumSteps = 60;
# dEpsilon = (Max_Epsilon - Min_Epsilon)/NumSteps
dEpsilon = 0.002;
perturbation_amplitude = 0.;
THRESH_WRINKLE = 1E-2

# geometry of the mesh
L=7.0;
W=0.565;
HD = 0.065
H=0.05;
Hu_box = 1.0; # expected maximum vertical displacement

# elastic parameter
E_f = 20.0;
E_d = 2.0
E_s =60.0;
nu_f = 0.3;
nu_s = 0.3;
nu_d = 0.3;

# initial growth
gg_f = 0.0;     
gg_s = 0.0;
gg_d = 0.0;

tol = 1E-14

E = Expression('x[1] >= -H - tol ? E_f : (x[1] >= -HD - tol ? E_d : E_s)', degree=0,
               tol=tol, E_f=E_f, E_s=E_s,E_d=E_d, H=H, HD = HD)
nu = Expression('x[1] >= -H - tol ? nu_f : (x[1] >= -HD - tol ? nu_d : nu_s)', degree=0,
                tol=tol, nu_f=nu_f, nu_s=nu_s, nu_d=nu_d, H=H, HD = HD)
mu = Expression('x[1] >= -H - tol ? mu_f : (x[1] >= -HD - tol ? mu_d : mu_s)', degree=0,
                tol=tol, mu_f=E_f/2/(1+nu_f), mu_s=E_s/2/(1+nu_s), mu_d=E_d/2/(1+nu_d), H=H, HD = HD)
Lambda = Expression('x[1] >= -H - tol ? lambda_f : (x[1] >= -HD - tol ? lambda_d : lambda_s)', degree=0,
                    tol=tol, lambda_f=E_f*nu_f/(1+nu_f)/(1-2*nu_f), lambda_s=E_s*nu_s/(1+nu_s)/(1-2*nu_s), lambda_d=E_d*nu_d/(1+nu_d)/(1-2*nu_d), H=H, HD = HD)

Gr = Expression('x[1] >= -H - tol ? gg_f : (x[1] >= -HD - tol ? gg_d : gg_s)', degree=0,
                tol=tol, H=H, HD = HD, gg_f = gg_f, gg_s = gg_s, gg_d = gg_d)

if not has_petsc():
    print("DOLFIN must be compiled at least with PETSc 3.6 to run this demo.")
    exit(0)
##################################################


##################################################
### 2. Import Mesh & Define Function Space
    
# Read mesh and refine once
mesh = Mesh("trilayer_long.xml")
# plot(mesh)
# mesh = refine(mesh)

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create solution, trial and test functions
u, du, v = Function(V), TrialFunction(V), TestFunction(V)
u0 = Function(V)
##################################################


##################################################
### 3. Define Box Region for Solving Optimization Problem
    
# The displacement u must be such that the current configuration
# doesn't escape the box [xmin, xmax] x [ymin, ymax]
constraint_u = Expression(("xmax-x[0]", "ymax-x[1]"), xmax=L, ymax=Hu_box, degree=1)
constraint_l = Expression(("xmin-x[0]", "ymin-x[1]"), xmin=0.0, ymin=-W, degree=1)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)
##################################################


##################################################
### 4. Set Boundary Conditions
    
# set BC
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0],0.);
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0],L);
def bottom_boundary(x, on_boundary):
    return on_boundary and near(x[1],-W);

def assemble_boundary_conditions(comp):
    #clamped boundary conditions (fix x and y directions)
    # bc_left  = DirichletBC(V, Constant((0, 0)), left_boundary)
    # bc_right = DirichletBC(V, Constant((comp, 0)), right_boundary)
    
    #clamped boundary conditions (fix only x direction, movement in y is unrestricted)
    bc_left  = DirichletBC(V.sub(0), Constant(0), left_boundary)
    bc_right = DirichletBC(V.sub(0), Constant(comp), right_boundary)
    
    #clamped boundary conditions (fix only y direction, movement in x is unrestricted)
    bc_bottom = DirichletBC(V.sub(1), Constant(0), bottom_boundary)
    return [bc_left, bc_right, bc_bottom]

bc=assemble_boundary_conditions(0.)

for b in bc:
    b.apply(u_min.vector())
    b.apply(u_max.vector())
##################################################


##################################################
### 5. Define Stress and Strain

def epsilon(u):
    Io = Identity(d)             # Identity tensor
    Fo = (Io + grad(u))          # Deformation gradient + growth
    Co = Fo.T*Fo                 # Right Cauchy-Green tensor
    return 0.5*(Co-Io);

def sigma(u):
    straino=epsilon(u);
    return Lambda*tr(straino)*Identity(d) + 2*mu*straino
##################################################


##################################################
### 6. Elastic Problem Formulation 
    
# Compressible neo-Hookean model
I = Identity(mesh.geometry().dim())
Ft = I + grad(u)     # total deformation tensor
Fg = as_matrix(((1. + Gr,0.),(0.,1.)))  # growth tensor
F = dot(Ft,inv(Fg))
C = F.T*F     # right Cauchy Green tensor
Ic = tr(C)
J  = det(F)
psi = (mu/2)*(Ic-2) -mu*ln(J)+(Lambda/2)*(ln(J))**2
  
# Surface force
# f = Constant((-0.08, 0.0))
    
# Variational formulation
elastic_energy = psi*dx
grad_elastic_energy = derivative(elastic_energy, u, v)
H_elastic_energy = derivative(grad_elastic_energy, u, du)
    
# stress
d = len(u)
VS = FunctionSpace(mesh, 'P', 1)
s = sigma(u) - (1./d)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises_temp = sqrt(3./2*inner(s, s))
von_Mises = project(von_Mises_temp, VS)
von_Mises.rename("von Mises","")

# strain
s = epsilon(u) - (1./d)*tr(epsilon(u))*Identity(d)  # deviatoric stress
deviatoric_strain_temp = sqrt(inner(s, s))
deviatoric_strain = project(deviatoric_strain_temp, VS)
deviatoric_strain.rename("deviatoric strain","")

# modulus
Eform = interpolate(E,VS)
Eform.rename("Modulus","")
##################################################


##################################################
### 7. Define Minimization problem 

class WrinklingProblem(OptimisationProblem):
    
    def __init__(self):
        OptimisationProblem.__init__(self)
    
    # Objective function
    def f(self, x):
        u.vector()[:] = x
        return assemble(elastic_energy)
    # Gradient of the objective function
    def F(self, b, x):
        u.vector()[:] = x
        assemble(grad_elastic_energy, tensor=b)
  
    # Hessian of the objective function
    def J(self, A, x):
        u.vector()[:] = x
        assemble(H_elastic_energy, tensor=A)
    
# Create the PETScTAOSolver
solver = PETScTAOSolver()
    
# Set some parameters
solver.parameters["method"] = "tron"
solver.parameters["monitor_convergence"] = True
solver.parameters["report"] = True
solver.parameters["maximum_iterations"] = 500

# Uncomment this line to see the available parameters
# info(parameters, True)
    
# Parse (PETSc) parameters
parameters.parse()
##################################################


##################################################
### 8. Setting Output Files Directions + Initial Conditions         

# outout file
file1 = File("test_trilayer 1/Es="+ str(E_s) + "/displacement.pvd");
file2 = File("test_trilayer 1/Es="+ str(E_s) + "/von_mises.pvd");
file3 = File("test_trilayer 1/Es="+ str(E_s) + "/dev_strain.pvd");
fileE = File("test_trilayer 1/Es="+ str(E_s) + "/modulus.pvd");
#filew = open("test_trilayer 1/wavelength_"+ str(E_s) +".txt",'w');

# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self, **kwargs):
        return
    def eval(self, values, x):
        values[0] = 0.0;
        values[1] = 0.0;
    def value_shape(self):
        return (2,)
    
# Class representing perterbuations
class RandomPerturbations(Expression):
    def __init__(self, **kwargs):
        return
    def eval(self, values, x):
        values[0] = 0.0;
        values[1] = perturbation_amplitude*(0.5 - random.random());
    def value_shape(self):
        return(2,)


u0.interpolate(InitialConditions(degree = 1)) # initial equilibrium position
u.vector()[:] = u0.vector()

Delta = Min_Epsilon    #initial growth
idx = 0
flag = 0
##################################################


##################################################
### 9. Main Loops Solving the Problem

while (Delta < Max_Epsilon) & (flag < 3):
    idx += 1
    Delta += dEpsilon
    Gr.gg_f = Delta;
    Gr.gg_d = Delta;
    
    # Solve the problem
    solver.solve(WrinklingProblem(), u.vector(), u_min.vector(), u_max.vector())
    
    # Save solution
    file1 << (u,Delta/(1+Delta));
    
    s = sigma(u) - (1./d)*tr(sigma(u))*Identity(d)  # deviatoric stress
    von_Mises_temp = sqrt(3./2*inner(s, s))
    von_Mises.vector()[:] = project(von_Mises_temp, VS).vector()
    
    file2 << (von_Mises,Delta/(1+Delta));
    
    s = epsilon(u) - (1./d)*tr(epsilon(u))*Identity(d)  # deviatoric strain
    deviatoric_strain_temp = sqrt(inner(s, s))
    deviatoric_strain.vector()[:] = project(deviatoric_strain_temp, VS).vector()
    file3 << (deviatoric_strain ,Delta/(1+Delta))
    
    fileE << (Eform, Delta/(1+Delta));
    
    # decide whether wrinkle or not
    temp_u = []
    for x_ in np.linspace(0.,L,701):
            temp_u.append(u(x_,0.)[1])
     
    if max(temp_u)- min(temp_u) > THRESH_WRINKLE:
        # record profile for wavelength analysis
#        for y_dp in temp_u:
#            filew.write(str(y_dp)+' ')
#        filew.write(str(Delta/(1+Delta))+ ' ' + str(L)+'\n')
        flag += 1
        print("current compressive strain is "+str(Delta/(1+Delta)))
    
    
    # Plot the current configuration
    dp = plot(u, mode="displacement", wireframe=True, title="Displacement field")
    plt.colorbar(dp)
    # sp = plot(von_Mises,title = "stress field")
    # plt.colorbar(sp)
    plt.show()
    
#filew.close()
##################################################

