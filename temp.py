from __future__ import print_function
from dolfin import *
import numpy as np
import datetime
import os
import math
set_log_level(30)
# Choose example: real or complex
example = "complex"

# Mesh parameters
dim     = 2
crossed = False             # For 2D only
length  = 2.4
width   = .3
res_x   = 8                 # Number of elements per wavelength along x-axis

# Complex example parameters
moden = 1.

# System properties
compressible = False
pcgls        = True
meas_sets    = 1
rom2         = 36.*math.pi**2.

# Material properties
mu_true  = 1.
mu_guessed = 0.95
mu_inc   = 0.
sigma_mu = length/8.
lambda_  = 1.e6

# Regularization and CG paramters
gamma = 1.0e-4             # Reg parameter
c_1   = 1.e-4              # (CG) Wolfe conditions parameters
c_2   = 0.1


# Iteration paramters
itprint = 1                # Print cost for every i-th iteration
itsave  = 5                # Save results for every i-th iteration
max_it  = 1000             # maximum number of iterations
tol     = 1.e-2            # tolerance for convergance (relative change in cost in 5 iterations)
delta   = 1.e-9            # small step length to determine Ap and alpha
    
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
h = length/nel_x
nel_y = int(round(width/h))
if dim == 2:
    if crossed == True:
        mesh = RectangleMesh(Point(0.,0.),Point(length,width),nel_x,nel_y, "crossed")
    else:
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
Vu    = V.sub(0).collapse()
Vmu   = V.sub(2).collapse()
mu_guess = interpolate(Expression('mu_1', mu_1 = mu_guessed, degree=1),Vmu)
mu_w_inc = interpolate(Expression('mu_0 + mu_1*exp(-pow(x[0] - le/2., 2.)/pow(sigma, 2.))',\
                                  mu_0 = mu_true, mu_1 = mu_inc, le = length, sigma = sigma_mu, degree = 2), Vmu)
mu_trial = TrialFunction(Vmu)
mu_test  = TestFunction(Vmu)
mu_temp, mu_old, mu_new = Function(Vmu), Function(Vmu), Function(Vmu)
ur_save, ui_save, ur2_save, ui2_save = Function(Vu), Function(Vu), Function(Vu), Function(Vu)
A_g = assemble(inner(mu_trial, mu_test)*dx)              # Mass matrix

# Define the problem    
class new_problem():
    def __init__(self, mu, *args):
        # Initializing measured dispalcements and parameters that depend on shear modulus
        self.mu = mu
        self.ks = sqrt(rom2/self.mu)
        self.sn = moden*math.pi/(self.ks*width)
        self.cn = sqrt(1.0-self.sn*self.sn)
        if args:
            self.umr = list(args)[0]
            self.umi = list(args)[1]
            if len(list(args)) > 2:
                self.umr2 = list(args)[2]
                self.umi2 = list(args)[3]
        if compressible:
            self.kappa = 1/(lambda_ + (2/3)*self.mu)
        else:
            self.kappa = 0
        if pcgls:
            c          = cos(self.ks*h)
            self.tau1  = h*h/self.mu/2.0
            self.tau2  = (((1+self.tau1*rom2)*(self.ks*self.ks*h*h/6.0)*(2.0+c)/(1.0-c)) - 1.0)/(rom2*self.ks*self.ks)
        else:
            self.tau1 = 0.
            self.tau2 = 0.
            
        # Boundary Conditions 
        if example == "real":
            if dim == 2:
                bclefta  = DirichletBC(V.sub(0), Constant((0., 0.)), left)
                bcleft   = DirichletBC(V.sub(0), Constant((0., .1)), left)
                bcrighta = DirichletBC(V.sub(0), Constant((0., 0.)), right)
                bcright  = DirichletBC(V.sub(0), Constant((0., .1)), right)
                bctopbot = DirichletBC(V.sub(0).sub(0), Constant(0.), topbot)
                self.bc  = [bcleft, bctopbot]
                self.bca = [bclefta, bctopbot]
            elif dim == 3:
                bclefta   = DirichletBC(V.sub(0), Constant((0., 0., 0.)), left)
                bcleft    = DirichletBC(V.sub(0), Constant((0., .1, 0.)), left)
                bctopbot2 = DirichletBC(V.sub(0).sub(2), Constant(0.), topbot)
                bctopbot  = DirichletBC(V.sub(0).sub(0), Constant(0.), topbot)
                bcbackfront = DirichletBC(V.sub(0).sub(2), Constant(0.), backfront)
                self.bc  = [bcleft, bctopbot, bctopbot2, bcbackfront]
                self.bca = [bclefta, bctopbot, bctopbot2, bcbackfront]
            else:
                raise ValueError("bcs(V, dim = 0)")
        elif example == "complex":
            self.ks_true = project(sqrt(rom2/mu_w_inc), Vmu)
            self.sn_true = project(moden*math.pi/(self.ks_true*width), Vmu)
            self.cn_true = project(sqrt(1.0-(self.sn_true*self.sn_true)), Vmu)
            if dim == 2:
                uexactr = Expression(("(-1.0*s/c*cos(m*pi*x[1]/height)*sin(c*k*x[0]))", "(sin(m*pi*x[1]/height)*cos(c*k*x[0]))"),\
                    m = moden, s = self.sn_true, c = self.cn_true, k = self.ks_true, height = width, degree=1)
                uexacti = Expression( ( "(s/c*cos(m*pi*x[1]/height)*cos(c*k*x[0]))", "(sin(m*pi*x[1]/height)*sin(c*k*x[0]))"),\
                    m = moden, s = self.sn_true, c = self.cn_true, k = self.ks_true, height = width, degree=1)
                # Enforce Dirichlet condition on y=x1 component of the displacement on top/bottom boundaries:
                bctbr = DirichletBC(V.sub(0).sub(1), zero, topbot)     # x1 or y component
                bctbi = DirichletBC(V.sub(1).sub(1), zero, topbot)     # x1 or y component
                # Enforce exact solution as Dirichlet condition on left boundary: 
                bclr = DirichletBC(V.sub(0), uexactr, left)
                bcli = DirichletBC(V.sub(1), uexacti, left)
                bclra= DirichletBC(V.sub(0), zeroo, left)
                bclia= DirichletBC(V.sub(1), zeroo, left)
                # Collect boundary conditions
                self.bc = [bctbr, bctbi, bclr,  bcli]
                self.bca= [bctbr, bctbi, bclra, bclia]
            elif dim == 3:
                uexactr = Expression(("(-1.0*s/c*cos(m*pi*x[1]/height)*sin(c*k*x[0]))", "(sin(m*pi*x[1]/height)*cos(c*k*x[0]))", "0.0"),\
                    m = moden, s = self.sn_true, c = self.cn_true, k = self.ks_true, height = width, degree=1)
                uexacti = Expression( ( "(s/c*cos(m*pi*x[1]/height)*cos(c*k*x[0]))", "(sin(m*pi*x[1]/height)*sin(c*k*x[0]))", "0.0"),\
                    m = moden, s = self.sn_true, c = self.cn_true, k = self.ks_true, height = width, degree=1)
                # Enforce Dirichlet condition on y=x1 component of the displacement on top/bottom boundaries:
                bctbr = DirichletBC(V.sub(0).sub(1), zero, topbot)     # x1 or y component
                bctbi = DirichletBC(V.sub(1).sub(1), zero, topbot)     # x1 or y component
                # Enforce exact solution as Dirichlet condition on left boundary: 
                bclr = DirichletBC(V.sub(0), uexactr, left)
                bcli = DirichletBC(V.sub(1), uexacti, left)
                bclra= DirichletBC(V.sub(0), zerooo, left)
                bclia= DirichletBC(V.sub(1), zerooo, left)
                # Enforce zero normal displacement as Dirichlet condition on front+back boundaries: 
                bcfbr = DirichletBC(V.sub(0).sub(2), zero, backfront)
                bcfbi = DirichletBC(V.sub(1).sub(2), zero, backfront)
                # Collect boundary conditions
                self.bc = [bctbr, bctbi, bclr,  bcli,  bcfbr, bcfbi]
                self.bca= [bctbr, bctbi, bclra, bclia, bcfbr, bcfbi]
            global ur_exact
            ur_exact =  project(uexactr, Vu)
            global ui_exact
            ui_exact =  project(uexacti, Vu)
            
    def state(self):
        ur_trial, ui_trial, pr_trial,   pi_trial = TrialFunctions(V)
        ur_test,  ui_test,  pr_test,    pi_test  = TestFunctions(V)
        if example == "complex":
            self.coeff_1 = 2.*self.ks*self.mu*self.cn
            self.coeff_2 = self.ks*self.mu*(self.cn*self.cn-self.sn*self.sn)/self.cn
            surface_int_r_s = (self.coeff_1*ur_test[0]*ui_trial[0] + self.coeff_2*ur_test[1]*ui_trial[1])*ds(1)
            surface_int_i_s = - (self.coeff_1*ui_test[0]*ur_trial[0] + self.coeff_2*ui_test[1]*ur_trial[1])*ds(1)
        elif example == "real":
            surface_int_r_s = 0.
            surface_int_i_s = 0.
        w  = Function(V)
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
        solve(a == L, w, self.bc)
        ur = w.split(deepcopy= True)[0]
        ui = w.split(deepcopy= True)[1]
        pr = w.split(deepcopy= True)[2]
        pi = w.split(deepcopy= True)[3]
        if meas_sets == 2:
            solve(a == L, w, self.bc2)
            ur2 = w.split(deepcopy= True)[0]
            ui2 = w.split(deepcopy= True)[1]
            pr2 = w.split(deepcopy= True)[2]
            pi2 = w.split(deepcopy= True)[3]
            return [ur, ui, pr, pi, ur2, ui2, pr2, pi2]
        return [ur, ui, pr, pi]
    
    def adjoint(self, ur, ui, *u2):
        ura_trial, uia_trial, pra_trial, pia_trial = TrialFunctions(V)
        ura_test,  uia_test,  pra_test,  pia_test  = TestFunctions(V)
        if example == "complex":
            surface_int_r_a = - (self.coeff_1*ura_test[0]*uia_trial[0] + self.coeff_2*ura_test[1]*uia_trial[1])*ds(1)
            surface_int_i_a = (self.coeff_1*uia_test[0]*ura_trial[0] + self.coeff_2*uia_test[1]*ura_trial[1])*ds(1)
        elif example == "real":
            surface_int_r_a = 0.
            surface_int_i_a = 0.
        w  = Function(V)
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
        L  = - (inner(ur - self.umr, ura_test) + inner(ui - self.umi, uia_test))*dx
        a  = ar + ai
        # Compute solution
        solve(a == L, w, self.bca)
        ura = w.split(deepcopy= True)[0]
        uia = w.split(deepcopy= True)[1]
        pra = w.split(deepcopy= True)[2]
        pia = w.split(deepcopy= True)[3]
        if meas_sets == 2:
            ur2  = list(u2)[0]
            ui2  = list(u2)[1]
            L2  = - (inner(ur2 - self.umr2, ura_test) + inner(ui2 - self.umi2, uia_test))*dx
            solve(a == L2, w, self.bca2)
            ura2 = w.split(deepcopy= True)[0]
            uia2 = w.split(deepcopy= True)[1]
            pra2 = w.split(deepcopy= True)[2]
            pia2 = w.split(deepcopy= True)[3]
            return [ura, uia, pra, pia, ura2, uia2, pra2, pia2]
        return [ura, uia, pra, pia]    
    
    def cost(self, ur, ui, *u_2):
        reg = 0.5 * gamma * assemble(inner(grad(self.mu), grad(self.mu))*dx)
        if meas_sets == 1:
            misfit = 0.5 *(assemble((inner(ur - self.umr, ur - self.umr) + inner(ui - self.umi, ui - self.umi))*dx))
        elif meas_sets == 2:
            if u_2:
                ur2  = list(u_2)[0]
                ui2  = list(u_2)[1]
                misfit = 0.5 *(assemble((inner(ur - self.umr, ur - self.umr) + inner(ui - self.umi, ui - self.umi)\
                                        + inner(ur2 - self.umr2, ur2 - self.umr2) + inner(ui2 - self.umi2, ui2 - self.umi2))*dx))
            else:
                raise ValueError("Displacements are missing from the second measured set")
        return [reg + misfit, misfit, reg]

    
    # Define gradient equation
    def gradient(self, ur, ui, ura, uia, *u_2):
        if example == "complex":
            dks_dmu   = - self.ks/(2.*self.mu)
            dsn_dmu   = self.sn/(2.*self.mu)
            dcn_dmu   = -(self.sn*self.sn/self.cn)/(2.*self.mu)
            d_coeff_1 = self.ks*(self.cn*self.cn-self.sn*self.sn)/self.cn
            d_coeff_2 = self.ks*(2.*self.cn - (1./(2.*self.cn)) - (1./(2.*self.cn*self.cn*self.cn)))
            surface_int_g = (mu_test*(d_coeff_1*(ura[0]*ui[0] - uia[0]*ur[0]) + d_coeff_2*(ura[1]*ui[1] - uia[1]*ur[1])))*ds(1)
        elif example == "real":
            surface_int_g = 0.
        g = Vector()
        A_g.init_vector(g, 0)
        if meas_sets == 1:
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
        
def find_alpha(mu, mu_dir, g_1):
    mu_temp.assign(mu)
    mu_temp.vector().axpy(delta, mu_dir)
    if meas_sets == 1:
        new_inst = new_problem(mu_temp, umr, umi)
        result_list = new_inst.get_cost_and_g()
        g_2 = result_list[5]
    elif meas_sets == 2:
        new_inst = new_problem(mu_temp, umr, umi, umr2, umi2)
        result_list = new_inst.get_cost_and_g()
        g_2 = result_list[7]
    g_2.axpy(-1., g_1)
    Ap = Vector(g_2)
    Ap.set_local(np.zeros(len(g_2)))
    Ap.axpy(1./delta, g_2)
    return [result_list, -(mu_dir.inner(g_1))/(mu_dir.inner(Ap)), Ap]
        
def CG_update(mu_old, g_old, mu_dir):
    gp = g_old.inner(mu_dir)
    mu_new.assign(mu_old)
    [result_list, alpha, Ap] = find_alpha(mu_old, mu_dir, g_old)
    mu_new.vector().axpy(alpha, mu_dir)
    if meas_sets == 1:
        new_inst = new_problem(mu_new, umr, umi)
    elif meas_sets == 2:
        new_inst  = new_problem(mu_new, umr, umi, umr2, umi2)
    result_list = new_inst.get_cost_and_g()
    result_list.append(gp)
    result_list.append(alpha)
    g_new = result_list[5]
    return result_list
    
def CG_step_one(mu_zero, g_zero):
    mu_dir = Vector(g_zero)
    mu_dir.set_local(np.zeros(len(g_zero)))
    mu_dir.axpy(-1, g_zero)
    result_list = CG_update(mu_zero, g_zero, mu_dir)
    result_list.append(mu_dir)
    return result_list

def update_mu_dir(mu_dir, g_new, mu_new):
    beta = (g_new.inner(g_new))/(g_old.inner(g_old))
    mu_dir *= beta
    mu_dir.axpy(-1, g_new)
    g_old.set_local(np.zeros(len(g_new)))
    g_old.axpy(1, g_new)
    mu_old.assign(mu_new)
    return [mu_dir, g_old, mu_old]

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
        
# Folder containing results
counter = 0
while os.path.exists("%s/%s" % (os.getcwd(), "results/%s_example_%2d"%(example,counter))):
    counter += 1
else:
    if os.path.exists("%s/%s" % (os.getcwd(), "results")):
        pass
    else:
        os.mkdir("%s/%s" % (os.getcwd(), "results"))
    dirname = "results/%s_example_%2d"%(example,counter)
    os.mkdir("%s/%s" % (os.getcwd(), dirname))
    os.mkdir("%s/%s/%s_%dD" % (os.getcwd(), dirname, subdirname, dim))
        
def itprint_func():
    if iteration % itprint == 0 :
        print("%4d %1s %1s %1s %1s %2.2e %1s %2.2e %1s %2.2e %1s %2.2e %1s %2.2e" % \
                  (iteration, sp, sp, sp, sp, cost_new, sp, misfit_new, sp, reg_new, sp,\
                   sqrt(g_new.inner(g_new))/float(len(g_new)), sp, alpha))
        
    if iteration % itsave == 0:
        File("%s/%s_%dD/iterations/strain12_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(grad(ur_new.split(deepcopy=True)[0])[1], Vmu)
        File("%s/%s_%dD/iterations/ur_%d.pvd" % (dirname, subdirname, dim, iteration)) << ur_new
        File("%s/%s_%dD/iterations/mu_%d.pvd" % (dirname, subdirname, dim, iteration)) << mu_new
        File("%s/%s_%dD/iterations/urdiff_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(ur_new - umr, Vu)
        File("%s/%s_%dD/iterations/grad_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(grad(mu_new), Vu)
        if meas_sets == 2:
            File("%s/%s_%dD/iterations/ur2_%d.pvd" % (dirname, subdirname, dim, iteration)) << ur2_new
            File("%s/%s_%dD/iterations/urdiff2_%d.pvd" % (dirname, subdirname, dim, iteration)) << project(ur2_new - umr2, Vu)
            
def write_to_file():
    txtfile  = open("%s/%s/%s_%dD/ReadMe.txt" % (os.getcwd(), dirname, subdirname, dim), "w")
    txtfile.write("This file was generated at %s\n\n\n" %(datetime.datetime.now()))
    txtfile.write("Here you can find some information about the solved problem:\n\n")
    txtfile.write("dimension = %d    crossed triangles mesh: %s\n" %(dim, str(crossed)))
    txtfile.write("length of domain : %2.1e    width of domain : %2.1e\n" %(length, width))
    txtfile.write("number of elements in the x-direction: %d    y-direction: %d\n" %(nel_x, nel_y))
    txtfile.write("PCGLS: %s    Compressible: %s    number of measured sets : %d\n" %(str(pcgls), str(compressible), meas_sets))
    txtfile.write("Background shear modulus: %2.2e    inclusion's modulus: %2.2e    initial guess for modulus: %2.2e\n" \
                  %(mu_true, mu_true+mu_inc, mu_guessed))
    txtfile.write("rho*omega**2: %2.2e    lambda: %2.2e\n" % (rom2 , lambda_))
    txtfile.write("Regularization parameter: %2.2e    Wolfe conditions constants c_1 and c_2: %2.2e  %2.2e\n" %(gamma, c_1, c_2))
    txtfile.write("Number of iterations: %d    norm(grad) at last iteration: %2.2e" %(iteration, sqrt(g_new.inner(g_new))/float(len(g_new))))

    txtfile.close()
    ur_diff = project(ur_new - umr, Vu)
    ui_diff = project(ui_new - umi, Vu)
    gradmu  = project(grad(mu_new), Vu)
    
    File("%s/%s_%dD/mu_real.pvd" % (dirname, subdirname, dim)) << mu_w_inc
    File("%s/%s_%dD/ur.pvd" % (dirname, subdirname, dim)) << ur_new
    File("%s/%s_%dD/ui.pvd" % (dirname, subdirname, dim)) << ui_new
    File("%s/%s_%dD/mu.pvd" % (dirname, subdirname, dim)) << mu_new
    #File("%s/pcgls_%dD/mu_real.pvd" % (dirname, dim)) << mu_w_inc
    File("%s/%s_%dD/urdiff.pvd" % (dirname, subdirname, dim)) << ur_diff
    File("%s/%s_%dD/uidiff.pvd" % (dirname, subdirname, dim)) << ui_diff
    File("%s/%s_%dD/grad_mu.pvd" % (dirname, subdirname, dim)) << gradmu
    if example == "complex":
        File("%s/%s_%dD/ur_exact.pvd" % (dirname, subdirname, dim)) << ur_exact
        File("%s/%s_%dD/ui_exact.pvd" % (dirname, subdirname, dim)) << ui_exact
    if meas_sets == 2:
        ur2_diff = project(ur2_new - umr2, Vu)
        ui2_diff = project(ui2_new - umi2, Vu)
        File("%s/%s_%dD/ur2.pvd" % (dirname, subdirname, dim)) << ur2_new
        File("%s/%s_%dD/ui2.pvd" % (dirname, subdirname, dim)) << ui2_new
        File("%s/%s_%dD/ur2diff.pvd" % (dirname, subdirname, dim)) << ur2_diff
        File("%s/%s_%dD/ui2diff.pvd" % (dirname, subdirname, dim)) << ui2_diff
            
measurements_inst = new_problem(mu_w_inc)
um_list = measurements_inst.state()
[umr, umi] = um_list[0:2]
File("%s/%s_%dD/ur_measured.pvd" % (dirname, subdirname, dim)) << umr
File("%s/%s_%dD/ui_measured.pvd" % (dirname, subdirname, dim)) << umi
if meas_sets == 2:
    [umr2, umi2] = um_list[4:6]
    File("%s/%s_%dD/ur2_measured.pvd" % (dirname, subdirname, dim)) << umr2
    File("%s/%s_%dD/ui2_measured.pvd" % (dirname, subdirname, dim)) << umi2

if meas_sets == 1:
    firstit_inst = new_problem(mu_guess, umr, umi)
    [ur_old, ui_old, cost_old, misfit_old, reg_old, g_old] = firstit_inst.get_cost_and_g()
elif meas_sets == 2:
    firstit_inst = new_problem(mu_guess, umr, umi, umr2, umi2)
    [ur_old, ui_old, ur2_old, ui2_old, cost_old, misfit_old, reg_old, g_old] = firstit_inst.get_cost_and_g()

mu_old.assign(mu_guess)
cost_list = [cost_old]

if meas_sets == 1:
    [ur_new, ui_new, cost_new, misfit_new, reg_new, g_new, gp, alpha, mu_dir] = CG_step_one(mu_old, g_old)
elif meas_sets == 2:
    [ur_new, ui_new, ur2_new, ui2_new, cost_new, misfit_new, reg_new, g_new, gp, alpha, mu_dir] = CG_step_one(mu_old, g_old)

sp = ""
iteration = 0
beta  = 0.
print("Iteration      cost      misfit      reg      ||grad||    alpha")
while iteration < max_it:
    itprint_func()
    iteration += 1
    cost_list.append(cost_new)
    if iteration > 5 and abs(cost_list[iteration] - cost_list[iteration-5])< tol*cost_list[iteration-5]:
        break
    
    if cost_new > cost_old + c_1*alpha*gp or abs(g_new.inner(mu_dir)) > -c_2*gp:
        print("Wolfe conditions not met")
            
    [mu_dir, g_old, mu_old] = update_mu_dir(mu_dir, g_new, mu_new)
    if meas_sets == 1:
        [ur_new, ui_new, cost_new, misfit_new, reg_new, g_new, gp, alpha] = CG_update(mu_old, g_old, mu_dir)
    elif meas_sets == 2:
        [ur_new, ui_new, ur2_new, ui2_new, cost_new, misfit_new, reg_new, g_new, gp, alpha] = CG_update(mu_old, g_old, mu_dir)
    
write_to_file()

print("finished with normalized norm(grad) = %2.2e" %(sqrt(g_new.inner(g_new))/float(len(g_new))))
