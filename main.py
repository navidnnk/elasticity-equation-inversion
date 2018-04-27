from __future__ import print_function
from dolfin import *
import numpy as np
import datetime
import os
import math
set_log_level(30)

# Choose example: real or complex
example    = "complex"
analytical = True          # Use analytical solution or the foward solution as simulated displacement data
noise_level = 0.0          

# Mesh parameters
dim     = 2                 # Dimension of geometry
length  = 2.              
width   = .3
res_x   = 8                # Number of elements per wavelength along x-axis

# Complex example parameters
moden = 1.

# System properties
compressible = False
pcgls        = True
meas_sets    = 1
rom2         = 36.*math.pi**2.

# Material properties
mu_true  = 1.               # Target shear modulus
mu_guessed = 0.95           # Initial shear modulus
mu_inc   = 0.               # Inclusion's shear modulus
sigma_mu = length/8.        # ~ width of inclusion
lambda_  = 1.e6             # First Lame modulus

# Iteration paramters
itprint = 1                # Print cost for every i-th iteration
itsave  = 5                # Save results for every i-th iteration
tol     = 1.e-3            # tolerance for convergance (relative change in cost in 5 iterations)
delta   = 1.e-9            # small step length to determine Ap and alpha
gamma = 1.0e-3             # Reg parameter
    
zero   = Constant(0.)
zeroo  = Constant((0., 0.))
zerooo = Constant((0., 0., 0.))                
if dim == 2:
    f = zeroo
elif dim == 3:
    f = zerooo

# Define boundaries
def topbot(x):
    return x[1] < DOLFIN_EPS or x[1] > width - DOLFIN_EPS

def top(x):
    return x[1] > width - DOLFIN_EPS

def bot(x):
    return x[1] < DOLFIN_EPS

def leftright(x):
    return x[0] < DOLFIN_EPS or x[0] > length - DOLFIN_EPS

def left(x, on_boundary):
    return x[0] < DOLFIN_EPS

def right(x, on_boundary):
    return x[0] > length - DOLFIN_EPS

def backfront(x):
    return x[2] < DOLFIN_EPS or x[2] > width - DOLFIN_EPS

# Generate mesh
# number of elements along x-axis = number of waves (using mu_guessed) per unit length * number of elements per wave
nel_x = int(round(res_x*length*sqrt(rom2/mu_guessed)/(2.*math.pi)))
h = length/nel_x                       # element length
nel_y = int(round(width/h))            # To make triangles with equal sides
max_it  = 1000             # maximum number of iterations
if dim == 2:
    mesh = RectangleMesh(Point(0.,0.),Point(length,width),nel_x,nel_y, "right")
    V_ele = VectorElement("P", mesh.ufl_cell(), 1, dim=2)
elif dim == 3:
    mesh = BoxMesh(Point(0.,0.,0.), Point(length,width,width), nel_x, nel_y, nel_y)
    V_ele = VectorElement("P", mesh.ufl_cell(), 1, dim=3)
else:
    raise ValueError("dim must be 2 or 3")

# Right boundary for boundary condition: 
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > length - DOLFIN_EPS
right = Right()
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
right.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
# Function spaces  
F_ele = FiniteElement("P", mesh.ufl_cell(), 1)
V     = FunctionSpace(mesh, MixedElement([V_ele, V_ele, F_ele, F_ele]))
Vu    = V.sub(0).collapse()       # Vector function space
Vmu   = V.sub(2).collapse()       # Scalar function space
mu_guess = interpolate(Expression('mu_1', mu_1 = mu_guessed, degree=1),Vmu)
mu_w_inc = interpolate(Expression('mu_0 + mu_1*exp(-pow(x[0] - le/2., 2.)/pow(sigma, 2.))',\
                                  mu_0 = mu_true, mu_1 = mu_inc, le = length, sigma = sigma_mu, degree = 2), Vmu)
mu_trial = TrialFunction(Vmu)     # Trial function for mu 
mu_test  = TestFunction(Vmu)      # Test function for mu
mu_temp, mu_old, mu_new = Function(Vmu), Function(Vmu), Function(Vmu)       # Used in find_alpha, iterations (x2)
A_g = assemble(inner(mu_trial, mu_test)*dx)     # Mass matrix

# Boundary Conditions 
if example == "real":
    # Move the left face, top and bottom x[0] = 0, right side traction free
    if dim == 2:
        uexactr  = Expression(("0.", "0.1*cos(k*x[0])"), k = 2.*pi, degree = 1)
        uexacti  = zeroo
        bclefta  = DirichletBC(V.sub(0), zeroo, left)
        bcleft   = DirichletBC(V.sub(0), Constant((0., .1)), left)
        bctopbot = DirichletBC(V.sub(0).sub(0), zero, topbot)
        bc  = [bcleft, bctopbot]
        bca = [bclefta, bctopbot]         # Substituted the left side bc for zero for adjoint equation
    elif dim == 3:
    # Move the left face, top and bottom x[0], x[2] = 0, right side traction free, front and back x[2] = 0
        uexactr   = Expression(("0.", "0.1*cos(k*x[0])", "0."), k = 2.*pi, degree = 1)
        uexacti   = zerooo
        bclefta   = DirichletBC(V.sub(0), zerooo, left)
        bcleft    = DirichletBC(V.sub(0), Constant((0., .1, 0.)), left)
        bctopbot2 = DirichletBC(V.sub(0).sub(2), zero, topbot)
        bctopbot  = DirichletBC(V.sub(0).sub(0), zero, topbot)
        bcbackfront = DirichletBC(V.sub(0).sub(2), zero, backfront)
        bc  = [bcleft, bctopbot, bctopbot2, bcbackfront]
        bca = [bclefta, bctopbot, bctopbot2, bcbackfront]          # Substituted the left side bc for zero for adjoint equation
elif example == "complex":
    # Move the left face, top and bottom x[1] = 0, right side robin BC in state and adjoint equations
    # Use exact k_s for the analytical solution, and the BC
    ks_true = project(sqrt(rom2/mu_w_inc), Vmu)
    sn_true = project(moden*pi/(ks_true*width), Vmu)
    cn_true = project(sqrt(1.0-(sn_true*sn_true)), Vmu)
    if dim == 2:
        uexactr = Expression(("(-1.0*s/c*cos(m*pi*x[1]/height)*sin(c*k*x[0]))", "(sin(m*pi*x[1]/height)*cos(c*k*x[0]))"),\
            m = moden, s = sn_true, c = cn_true, k = ks_true, height = width, degree=1)
        uexacti = Expression( ( "(s/c*cos(m*pi*x[1]/height)*cos(c*k*x[0]))", "(sin(m*pi*x[1]/height)*sin(c*k*x[0]))"),\
            m = moden, s = sn_true, c = cn_true, k = ks_true, height = width, degree=1)
        bctbr = DirichletBC(V.sub(0).sub(1), zero, topbot)
        bctbi = DirichletBC(V.sub(1).sub(1), zero, topbot) 
        bclr = DirichletBC(V.sub(0), uexactr, left)
        bcli = DirichletBC(V.sub(1), uexacti, left)
        bclra= DirichletBC(V.sub(0), zeroo, left)
        bclia= DirichletBC(V.sub(1), zeroo, left)
        bc = [bctbr, bctbi, bclr,  bcli]
        bca= [bctbr, bctbi, bclra, bclia]           # Substituted the left side bc for zero for adjoint equation
    elif dim == 3:
        # Move the left face, top and bottom x[1] = 0, front and back x[2] = 0
        uexactr = Expression(("(-1.0*s/c*cos(m*pi*x[1]/height)*sin(c*k*x[0]))", "(sin(m*pi*x[1]/height)*cos(c*k*x[0]))", "0.0"),\
            m = moden, s = sn_true, c = cn_true, k = ks_true, height = width, degree=1)
        uexacti = Expression( ( "(s/c*cos(m*pi*x[1]/height)*cos(c*k*x[0]))", "(sin(m*pi*x[1]/height)*sin(c*k*x[0]))", "0.0"),\
            m = moden, s = sn_true, c = cn_true, k = ks_true, height = width, degree=1)
        bctbr = DirichletBC(V.sub(0).sub(1), zero, topbot)
        bctbi = DirichletBC(V.sub(1).sub(1), zero, topbot)
        bclr = DirichletBC(V.sub(0), uexactr, left)
        bcli = DirichletBC(V.sub(1), uexacti, left)
        bclra= DirichletBC(V.sub(0), zerooo, left)
        bclia= DirichletBC(V.sub(1), zerooo, left)
        bcfbr = DirichletBC(V.sub(0).sub(2), zero, backfront)
        bcfbi = DirichletBC(V.sub(1).sub(2), zero, backfront)
        bc = [bctbr, bctbi, bclr,  bcli,  bcfbr, bcfbi]
        bca= [bctbr, bctbi, bclra, bclia, bcfbr, bcfbi]           # Substituted the left side bc for zero for adjoint equation

# Define the problem    
class new_problem():
    def __init__(self, mu):
        # Initializing parameters that depend on shear modulus
        self.mu = mu
        self.ks = sqrt(rom2/self.mu)
        self.sn = moden*math.pi/(self.ks*width)
        self.cn = sqrt(1.0-self.sn*self.sn)
        if compressible:
            self.kappa = 1/(lambda_ + (2/3)*self.mu)      # Inverse of bulk modulus
        else:
            self.kappa = 0
        if pcgls:        # Initialize the pcgls coefficients
            c          = cos(self.ks*h)
            self.tau1  = h*h/self.mu/2.0
            self.tau2  = (((1+self.tau1*rom2)*(self.ks*self.ks*h*h/6.0)*(2.0+c)/(1.0-c)) - 1.0)/(rom2*self.ks*self.ks)
        else:
            self.tau1 = 0.
            self.tau2 = 0.
    # Define the state equation (forward problem)        
    def state(self):
        ur_trial, ui_trial, pr_trial, pi_trial = TrialFunctions(V)
        ur_test,  ui_test,  pr_test,  pi_test  = TestFunctions(V)
        if example == "complex":                # Surface integrals related to the robin BC
            self.coeff_1 = 2.*self.ks*self.mu*self.cn
            self.coeff_2 = self.ks*self.mu*(self.cn*self.cn-self.sn*self.sn)/self.cn
            surface_int_r_s = (self.coeff_1*ur_test[0]*ui_trial[0] + self.coeff_2*ur_test[1]*ui_trial[1])*ds(1)
            surface_int_i_s = - (self.coeff_1*ui_test[0]*ur_trial[0] + self.coeff_2*ui_test[1]*ur_trial[1])*ds(1)
        elif example == "real":
            surface_int_r_s = 0.
            surface_int_i_s = 0.
        w  = Function(V)
        # Left hand side of a(u, v) = l(v) with real and imaginary contributions decoupled
        ar = (inner(grad(ur_test), self.mu*(grad(ur_trial) + grad(ur_trial).T)) \
            - 2./3.*self.mu * div(ur_test) * div(ur_trial)\
            - inner(rom2*ur_test, ur_trial) - inner(pr_test, div(ur_trial)) - inner(pr_trial, div(ur_test))  \
            - inner(pr_test, self.kappa * pr_trial) \
            - self.tau1*inner(-grad(pr_test) + rom2*ur_test, -grad(pr_trial) + rom2*ur_trial) \
            + 0.5*self.tau2*rom2*rom2*inner(grad(ur_test) - grad(ur_test).T, grad(ur_trial) - grad(ur_trial).T ))*dx + surface_int_r_s
        ai = (inner(grad(ui_test), self.mu*(grad(ui_trial) + grad(ui_trial).T)) \
            - 2./3.*self.mu * div(ui_test) * div(ui_trial)\
            - inner(rom2*ui_test, ui_trial) - inner(pi_test, div(ui_trial)) - inner(pi_trial, div(ui_test))  \
            - inner(pi_test, self.kappa * pi_trial) \
            - self.tau1*inner(-grad(pi_test) + rom2*ui_test, -grad(pi_trial) + rom2*ui_trial) \
            + 0.5*self.tau2*rom2*rom2*inner(grad(ui_test) - grad(ui_test).T, grad(ui_trial) - grad(ui_trial).T ))*dx + surface_int_i_s            
        L  = inner(f, ur_test + self.tau1*grad(pr_test))*dx
        a  = ar + ai
        # Compute solution
        solve(a == L, w, bc)
        ur = w.split(deepcopy= True)[0]       # Split the real and imaginary components of displacement and the pressure
        ui = w.split(deepcopy= True)[1]
        pr = w.split(deepcopy= True)[2]
        pi = w.split(deepcopy= True)[3]
        if meas_sets == 2:
            solve(a == L, w, bc2)
            ur2 = w.split(deepcopy= True)[0]
            ui2 = w.split(deepcopy= True)[1]
            pr2 = w.split(deepcopy= True)[2]
            pi2 = w.split(deepcopy= True)[3]
            return [ur, ui, pr, pi, ur2, ui2, pr2, pi2]
        return [ur, ui, pr, pi]
    # Define the adjoint equation 
    def adjoint(self, ur, ui, *u2):
        ura_trial, uia_trial, pra_trial, pia_trial = TrialFunctions(V)
        ura_test,  uia_test,  pra_test,  pia_test  = TestFunctions(V)
        if example == "complex":              # Surface integrals related to the robin BC
            surface_int_r_a = - (self.coeff_1*ura_test[0]*uia_trial[0] + self.coeff_2*ura_test[1]*uia_trial[1])*ds(1)
            surface_int_i_a = (self.coeff_1*uia_test[0]*ura_trial[0] + self.coeff_2*uia_test[1]*ura_trial[1])*ds(1)
        elif example == "real":
            surface_int_r_a = 0.
            surface_int_i_a = 0.
        w  = Function(V)
        # Left hand side of a(u, v) = l(v) with real and imaginary contributions decoupled
        ar = (inner(grad(ura_test), self.mu*(grad(ura_trial) + grad(ura_trial).T)) \
            - 2./3.*self.mu * div(ura_test) * div(ura_trial) \
            - inner(rom2*ura_test, ura_trial) - inner(pra_test, div(ura_trial)) - inner(pra_trial, div(ura_test))  \
            - inner(pra_test, self.kappa * pra_trial)\
            - self.tau1*inner(-grad(pra_test) + rom2*ura_test, -grad(pra_trial) + rom2*ura_trial) \
            + 0.5*self.tau2*rom2*rom2*inner(grad(ura_test) - grad(ura_test).T, grad(ura_trial) - grad(ura_trial).T ))*dx + surface_int_r_a
        ai = (inner(grad(uia_test), self.mu*(grad(uia_trial) + grad(uia_trial).T)) \
            - 2./3.*self.mu * div(uia_test) * div(uia_trial) \
            - inner(rom2*uia_test, uia_trial) - inner(pia_test, div(uia_trial)) - inner(pia_trial, div(uia_test))  \
            - inner(pia_test, self.kappa * pia_trial)\
            - self.tau1*inner(-grad(pia_test) + rom2*uia_test, -grad(pia_trial) + rom2*uia_trial) \
            + 0.5*self.tau2*rom2*rom2*inner(grad(uia_test) - grad(uia_test).T, grad(uia_trial) - grad(uia_trial).T ))*dx + surface_int_i_a
        L  = - (inner(ur - umr, ura_test) + inner(ui - umi, uia_test))*dx
        a  = ar + ai
        # Compute solution
        solve(a == L, w, bca)
        ura = w.split(deepcopy= True)[0]       # Split the real and imaginary components of displacement and the pressure
        uia = w.split(deepcopy= True)[1]
        pra = w.split(deepcopy= True)[2]
        pia = w.split(deepcopy= True)[3]
        if meas_sets == 2:
            ur2  = list(u2)[0]
            ui2  = list(u2)[1]
            L2  = - (inner(ur2 - umr2, ura_test) + inner(ui2 - umi2, uia_test))*dx
            solve(a == L2, w, bca2)
            ura2 = w.split(deepcopy= True)[0]
            uia2 = w.split(deepcopy= True)[1]
            pra2 = w.split(deepcopy= True)[2]
            pia2 = w.split(deepcopy= True)[3]
            return [ura, uia, pra, pia, ura2, uia2, pra2, pia2]
        return [ura, uia, pra, pia]    
    # Define the cost function
    def cost(self, ur, ui, *u_2):
        reg = 0.5 * gamma * assemble(inner(grad(self.mu), grad(self.mu))*dx)       # Define regularization (currently H1)
        if meas_sets == 1:
            misfit = 0.5 *(assemble((inner(ur - umr, ur - umr) + inner(ui - umi, ui - umi))*dx))       # Define displacement mismatch
        elif meas_sets == 2:
            ur2  = list(u_2)[0]
            ui2  = list(u_2)[1]
            misfit = 0.5 *(assemble((inner(ur - umr, ur - umr) + inner(ui - umi, ui - umi)\
                                    + inner(ur2 - umr2, ur2 - umr2) + inner(ui2 - umi2, ui2 - umi2))*dx))
        return [reg + misfit, misfit, reg]
    # Define gradient equation
    def gradient(self, ur, ui, ura, uia, *u_2):
        if example == "complex":                # Derivatives of surface integrals related to the robin BC
            dks_dmu   = - self.ks/(2.*self.mu)
            dsn_dmu   = self.sn/(2.*self.mu)
            dcn_dmu   = -(self.sn*self.sn/self.cn)/(2.*self.mu)
            d_coeff_1 = self.coeff_2/self.mu
            d_coeff_2 = (self.coeff_2/(2.*self.mu)) - self.ks*(2. + (1/(self.cn*self.cn)))*(self.sn*self.sn/(2.*self.cn))
            surface_int_g = (mu_test*(d_coeff_1*(ura[0]*ui[0] - uia[0]*ur[0]) + d_coeff_2*(ura[1]*ui[1] - uia[1]*ur[1])))*ds(1)
        elif example == "real":
            surface_int_g = 0.
        g = Vector()
        A_g.init_vector(g, 0)       # Initialize the size of the gradient vector
        if meas_sets == 1:          # Right hand side of A_g*g=b
            b_g = assemble((inner(grad(ura), mu_test*(grad(ur) + grad(ur).T)) - 2./3.* mu_test * div(ura) * div(ur)\
                            + inner(grad(uia), mu_test*(grad(ui) + grad(ui).T)) - 2./3.* mu_test * div(uia) * div(ui)\
                            + inner(gamma*grad(self.mu), grad(mu_test)) )* dx + surface_int_g ) 
        elif meas_sets == 2:
            ur2  = list(u_2)[0]
            ui2  = list(u_2)[1]
            ura2 = list(u_2)[2]
            uia2 = list(u_2)[3]
            b_g = assemble((inner(grad(ura), mu_test*(grad(ur) + grad(ur).T)) - 2./3.* mu_test * div(ura) * div(ur)\
                            + inner(grad(ura2), mu_test*(grad(ur2) + grad(ur2).T)) - 2./3.* mu_test * div(ura2) * div(ur2)\
                            + inner(grad(uia), mu_test*(grad(ui) + grad(ui).T)) - 2./3.* mu_test * div(uia) * div(ui)\
                            + inner(grad(uia2), mu_test*(grad(ui2) + grad(ui2).T)) - 2./3.* mu_test * div(uia2) * div(ui2)\
                            + inner(gamma*grad(self.mu), grad(mu_test)) )* dx)
        solve(A_g, g, b_g)
        return g
    # Run one iteration of state >> cost & adjoint >> gradient with one function call
    def get_cost_and_g(self):
        if meas_sets == 1:
            [ur, ui, pr, pi] = self.state()
            [ura, uia, pra, pia] = self.adjoint(ur, ui)
            [cost_total, misfit, reg] = self.cost(ur, ui)
            g = self.gradient(ur, ui, ura, uia)
            return [ur, ui, cost_total, misfit, reg, g]
        elif meas_sets == 2:
            [ur, ui, pr, pi, ur2, ui2, pr2, pi2] = self.state()
            [ura,  uia,  pra,  pia, ura2, uia2, pra2, pia2]  = self.adjoint(ur, ui, ur2, ui2)
            [cost_total, misfit, reg] = self.cost(ur, ui, ur2, ui2)
            g = self.gradient(ur, ui, ura, uia, ur2, ui2, ura2, uia2)
            return [ur, ui, ur2, ui2, cost_total, misfit, reg, g]
# Line search for CG: alpha = -(p.g_old)/(p.Ap) where Ap = (g_temp - g)/delta where g_temp was calculated with mu + delta.p      
def find_alpha(mu, mu_dir, g_1):
    mu_temp.assign(mu)
    mu_temp.vector().axpy(delta, mu_dir)
    new_inst = new_problem(mu_temp)
    result_list = new_inst.get_cost_and_g()
    if meas_sets == 1:
        g_2 = result_list[5]
    elif meas_sets == 2:
        g_2 = result_list[7]
    g_2.axpy(-1., g_1)
    Ap = Vector(g_2)
    Ap.set_local(np.zeros(len(g_2)))
    Ap.axpy(1./delta, g_2)
    return -(mu_dir.inner(g_1))/(mu_dir.inner(Ap))
# Move mu one step (alpha) along the mu_dir direction       
def CG_update(mu_old, g_old, mu_dir):
    gp = g_old.inner(mu_dir)
    mu_new.assign(mu_old)
    alpha = find_alpha(mu_old, mu_dir, g_old)
    mu_new.vector().axpy(alpha, mu_dir)
    new_inst = new_problem(mu_new)
    result_list = new_inst.get_cost_and_g()
    result_list.append(gp)
    result_list.append(alpha)
    return result_list
# Set initial gradient as the first mu_dir to update mu    
def CG_step_one(mu_zero, g_zero):
    mu_dir = Vector(g_zero)
    mu_dir.set_local(np.zeros(len(g_zero)))
    mu_dir.axpy(-1, g_zero)
    result_list = CG_update(mu_zero, g_zero, mu_dir)
    result_list.append(mu_dir)
    return result_list
# Update the mu_dir, which is used to update mu
def update_mu_dir(mu_dir, g_new, mu_new):
    beta = (g_new.inner(g_new))/(g_old.inner(g_old))
    mu_dir *= beta
    mu_dir.axpy(-1, g_new)
    g_old.set_local(np.zeros(len(g_new)))
    g_old.axpy(1, g_new)
    mu_old.assign(mu_new)
    return [mu_dir, g_old, mu_old]
# Create results folder's title
if pcgls:
    if compressible:
        subdirname = "pcgls_compressible"
    else:
        subdirname = "pcgls_incompressible"
else:
    if compressible:
        subdirname = "no_pcgls_compressible"
    else:
        subdirname = "no_pcgls_incompressible"
        
# Create a folder that contains the results
counter = 0
while os.path.exists("%s/%s" % (os.getcwd(), "results/%s_example_%3d"%(example,counter))):
    counter += 1
else:
    if os.path.exists("%s/%s" % (os.getcwd(), "results")):
        pass
    else:
        os.mkdir("%s/%s" % (os.getcwd(), "results"))
    dirname = "results/%s_example_%3d"%(example,counter)
    os.mkdir("%s/%s" % (os.getcwd(), dirname))
    os.mkdir("%s/%s/%s_%dD" % (os.getcwd(), dirname, subdirname, dim))
# Print and save the results at every itprint and itsave iteration, respectively      
def it_func():
    if iteration % itprint == 0 :
        print("%4d %1s %1s %1s %1s %2.2e %1s %2.2e %1s %2.2e %1s %2.2e %1s %2.2e" % \
                  (iteration, sp, sp, sp, sp, cost_new, sp, misfit_new, sp, reg_new, sp,\
                   sqrt(g_new.inner(g_new))/float(len(g_new)), sp, alpha))
    if iteration % itsave == 0:
        File("%s/%s_%dD/iterations/ur_%d.pvd" % (dirname, subdirname, dim, iteration)) << ur_new
        File("%s/%s_%dD/iterations/ui_%d.pvd" % (dirname, subdirname, dim, iteration)) << ui_new
        File("%s/%s_%dD/iterations/mu_%d.pvd" % (dirname, subdirname, dim, iteration)) << mu_new
        File("%s/%s_%dD/iterations/urdiff_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(ur_new - umr, Vu)
        File("%s/%s_%dD/iterations/uidiff_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(ui_new - umi, Vu)
        if meas_sets == 2:
            File("%s/%s_%dD/iterations/ur2_%d.pvd" % (dirname, subdirname, dim, iteration)) << ur2_new
            File("%s/%s_%dD/iterations/ui2_%d.pvd" % (dirname, subdirname, dim, iteration)) << ui2_new
            File("%s/%s_%dD/iterations/urdiff2_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(ur2_new - umr2, Vu)
            File("%s/%s_%dD/iterations/uidiff2_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(ui2_new - umi2, Vu)
# Write the information about the problem into a text file and save all the final results            
def write_to_file():
    txtfile  = open("%s/%s/%s_%dD/ReadMe.txt" % (os.getcwd(), dirname, subdirname, dim), "w")
    txtfile.write("This file was generated at %s\n\n\n" %(datetime.datetime.now()))
    txtfile.write("Here you can find some information about the solved problem:\n\n")
    txtfile.write("dimension = %d    length of domain : %2.1e    width of domain : %2.1e\n" %(dim, length, width))
    txtfile.write("number of elements in the x-direction: %d    y-direction: %d\n" %(nel_x, nel_y))
    txtfile.write("PCGLS: %s    Compressible: %s    number of measured sets : %d    Noise level: %2.2e\n" %(str(pcgls), str(compressible), meas_sets, noise_level))
    txtfile.write("Background shear modulus: %2.2e    inclusion's modulus: %2.2e    initial guess for modulus: %2.2e\n" \
                  %(mu_true, mu_true+mu_inc, mu_guessed))
    txtfile.write("rho*omega**2: %2.2e    lambda: %2.2e    Regularization parameter: %2.2e\n" % (rom2 , lambda_, gamma))
    txtfile.write("Number of iterations: %d    norm(grad) at last iteration: %2.2e" %(iteration, sqrt(g_new.inner(g_new))/float(len(g_new))))
    txtfile.close()
    File("%s/%s_%dD/mu_real.pvd" % (dirname, subdirname, dim)) << mu_w_inc
    File("%s/%s_%dD/ur_measured.pvd" % (dirname, subdirname, dim)) << umr
    File("%s/%s_%dD/ui_measured.pvd" % (dirname, subdirname, dim)) << umi
    File("%s/%s_%dD/ur.pvd" % (dirname, subdirname, dim)) << ur_new
    File("%s/%s_%dD/ui.pvd" % (dirname, subdirname, dim)) << ui_new
    File("%s/%s_%dD/mu.pvd" % (dirname, subdirname, dim)) << mu_new
    File("%s/%s_%dD/urdiff.pvd" % (dirname, subdirname, dim)) << project(ur_new - umr, Vu)
    File("%s/%s_%dD/uidiff.pvd" % (dirname, subdirname, dim)) << project(ui_new - umi, Vu)
    File("%s/%s_%dD/grad_mu.pvd" % (dirname, subdirname, dim)) << project(grad(mu_new), Vu)
    File("%s/%s_%dD/strainr.pvd" % (dirname, subdirname, dim)) << project(0.5*grad(umr.split(deepcopy=True)[0])[1] + 0.5*grad(umr.split(deepcopy=True)[0])[1], Vmu)
    File("%s/%s_%dD/straini.pvd" % (dirname, subdirname, dim)) << project(0.5*grad(umi.split(deepcopy=True)[0])[1] + 0.5*grad(umi.split(deepcopy=True)[0])[1], Vmu)
    if meas_sets == 2:
        ur2_diff = project(ur2_new - umr2, Vu)
        ui2_diff = project(ui2_new - umi2, Vu)
        File("%s/%s_%dD/ur2.pvd" % (dirname, subdirname, dim)) << ur2_new
        File("%s/%s_%dD/ui2.pvd" % (dirname, subdirname, dim)) << ui2_new
        File("%s/%s_%dD/ur2diff.pvd" % (dirname, subdirname, dim)) << ur2_diff
        File("%s/%s_%dD/ui2diff.pvd" % (dirname, subdirname, dim)) << ui2_diff
# Decide between using the analytical solution for displacement data and the solution of the forward problem using exact mu
if analytical:
    umr = interpolate(uexactr, Vu)
    umi = interpolate(uexacti, Vu)
else:
    measurements_inst = new_problem(mu_w_inc)
    um_list = measurements_inst.state()
    [umr, umi] = um_list[0:2]
    File("%s/%s_%dD/ur_measured.pvd" % (dirname, subdirname, dim)) << umr
    File("%s/%s_%dD/ui_measured.pvd" % (dirname, subdirname, dim)) << umi
    if meas_sets == 2:
        [umr2, umi2] = um_list[4:6]
        File("%s/%s_%dD/ur2_measured.pvd" % (dirname, subdirname, dim)) << umr2
        File("%s/%s_%dD/ui2_measured.pvd" % (dirname, subdirname, dim)) << umi2
# Add random noise as big as noise_level * max(um)
MAXr = umr.vector().norm("linf")
noiser = Vector(umr.vector())
noiser.set_local(noise_level * MAXr * np.random.normal(0, 1, len(umr.vector().get_local())))
umr.vector().axpy(1., noiser)
MAXi = umi.vector().norm("linf")
noisei = Vector(umr.vector())
noisei.set_local(noise_level * MAXi * np.random.normal(0, 1, len(umi.vector().get_local())))
umi.vector().axpy(1., noisei)
# Run the first iteration
mu_old.assign(mu_guess)
firstit_inst = new_problem(mu_old)
if meas_sets == 1:
    [ur_old, ui_old, cost_old, misfit_old, reg_old, g_old] = firstit_inst.get_cost_and_g()
    [ur_new, ui_new, cost_new, misfit_new, reg_new, g_new, gp, alpha, mu_dir] = CG_step_one(mu_old, g_old)
elif meas_sets == 2:
    [ur_old, ui_old, ur2_old, ui2_old, cost_old, misfit_old, reg_old, g_old] = firstit_inst.get_cost_and_g()
    [ur_new, ui_new, ur2_new, ui2_new, cost_new, misfit_new, reg_new, g_new, gp, alpha, mu_dir] = CG_step_one(mu_old, g_old)
cost_change = []     # Save the change in cost at every step

sp = ""
iteration = 0
max_it  = len(mesh.coordinates())       # maximum number of iterations
beta  = 0.
print("Starting iterative inversion...\nThe shear modulus will update for maximum %d iterations, equal to the number of nodes,"%(max_it))
print("or until relative cost change remains within %.1f%% of the last iteration cost for 5 iterations.\n"%(tol*100))
print("Iteration      cost      misfit      reg      ||grad||    alpha")
while iteration < max_it:
    it_func()
    iteration += 1
    cost_change.append(abs(cost_new - cost_old))
    if iteration > 5 and all(i < tol*cost_new for i in cost_change[-5:]):
        print("Converged before running max iterations.")
        break
            
    [mu_dir, g_old, mu_old] = update_mu_dir(mu_dir, g_new, mu_new)
    cost_old = cost_new
    if meas_sets == 1:
        [ur_new, ui_new, cost_new, misfit_new, reg_new, g_new, gp, alpha] = CG_update(mu_old, g_old, mu_dir)
    elif meas_sets == 2:
        [ur_new, ui_new, ur2_new, ui2_new, cost_new, misfit_new, reg_new, g_new, gp, alpha] = CG_update(mu_old, g_old, mu_dir)
else:
    print("Did not converge within %d iterations"%(max_it))
    
write_to_file()
