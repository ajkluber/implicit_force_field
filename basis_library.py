import time
import numpy as np
import sympy

import scipy.interpolate

import simtk.unit as unit

import simulation.openmm as sop

def polymer_library(n_beads, bonds=True, angles=True, non_bond_wca=True, non_bond_gaussians=True):
    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params 

    # remove units from paramters
    sigma_ply_nm = sigma_ply/unit.nanometer
    r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer

    one_half = sympy.Rational(1,2)

    # define all variables of the system symbolically
    max_n_args = 3*3
    xyz_sym = []
    for i in range(max_n_args/3):
        x_i = sympy.symbols('x' + str(i + 1))
        y_i = sympy.symbols('y' + str(i + 1))
        z_i = sympy.symbols('z' + str(i + 1))
        xyz_sym.append([x_i, y_i, z_i])
    x1, y1, z1 = xyz_sym[0]
    x2, y2, z2 = xyz_sym[1]
    x3, y3, z3 = xyz_sym[2]

    r12_sym = sympy.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r23_sym = sympy.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
    r13_sym = sympy.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)

    rij_args = (x1, y1, z1, x2, y2, z2)

    theta_ijk_sym = sympy.acos((r12_sym**2 + r23_sym**2 - r13_sym**2)/(2*r12_sym*r23_sym))
    theta_ijk_args = (x1, y1, z1, x2, y2, z2, x3, y3, z3)

    # calculate gradient with respect to each coordinate
    # for each function have list of participating coordinates.
    
    # scale force functions to approximate magnitude of parameters. Then output
    # coeff will all be around 1. This reduces the condition number
    kb_scale = sympy.Rational(300000,1)
    ka_scale = sympy.Rational(500,1)
    eps_scale = sympy.Rational(10, 17)
    gauss_scale = sympy.Rational(1,10)
    scale_factors = np.array([ float(x) for x in [kb_scale.evalf(),
        ka_scale.evalf(), eps_scale.evalf(), gauss_scale.evalf()]])

    dU_funcs = []
    dU_idxs = []
    dU_d_arg = []
    dU_dxi = []
    dU_ck = []

    if bonds:
        dU_bond_dxi = []
        dU_bond_ck = []
        dU_bond = []
        dU_bond_d_arg = []
        dbond_idxs = []
        for i in range(n_beads - 1):
            xi_idxs = np.arange(6) + i*3
            for n in range(len(rij_args)):
                dbond_idxs.append(xi_idxs)
                dU_bond_d_arg.append(n)
                dU_bond_dxi.append(xi_idxs[n])
                dU_bond_ck.append(len(dU_funcs))
                if i == 0:
                    # take derivative w.r.t. argument n
                    bond_func = kb_scale*one_half*(r12_sym - r0_nm)**2 # scaled
                    d_bond_func = -bond_func.diff(rij_args[n])
                    dU_bond.append(sympy.lambdify(rij_args, d_bond_func, modules="numpy"))
        dU_funcs.append(dU_bond)
        dU_idxs += dbond_idxs
        dU_d_arg += dU_bond_d_arg
        dU_dxi += dU_bond_dxi
        dU_ck += dU_bond_ck

    if angles:
        # angle potential
        dU_angle = []
        dU_angle_dxi = []
        dU_angle_ck = []
        dU_angle_d_arg = []
        dang_idxs = []
        for i in range(n_beads - 2):
            xi_idxs = np.arange(9) + i*3
            for n in range(len(theta_ijk_args)):
                dang_idxs.append(xi_idxs)
                dU_angle_dxi.append(xi_idxs[n])
                dU_angle_ck.append(len(dU_funcs))
                dU_angle_d_arg.append(n)
                if i == 0:
                    ang_func = ka_scale*one_half*(theta_ijk_sym - theta0_rad)**2  # scaled
                    d_ang_func = -ang_func.diff(theta_ijk_args[n])
                    dU_angle.append(sympy.lambdify(theta_ijk_args, d_ang_func, modules="numpy"))
        dU_funcs.append(dU_angle)
        dU_idxs += dang_idxs
        dU_d_arg += dU_angle_d_arg
        dU_dxi += dU_angle_dxi
        dU_ck += dU_angle_ck

    if non_bond_wca:
        # pairwise potential
        bond_cutoff = 3
        dU_pair = []
        dU_pair_dxi = []
        dU_pair_ck = []
        dU_pair_d_arg = []
        dpair_idxs = []
        for i in range(n_beads - bond_cutoff - 1):
            idxs1 = np.arange(3) + i*3
            for j in range(i + bond_cutoff + 1, n_beads):
                idxs2 = np.arange(3) + j*3
                xi_idxs = np.concatenate([idxs1, idxs2])
                for n in range(len(rij_args)):
                    dpair_idxs.append(xi_idxs)
                    dU_pair_dxi.append(xi_idxs[n])
                    dU_pair_ck.append(len(dU_funcs))
                    dU_pair_d_arg.append(n)
                    if (i == 0) and (j == (bond_cutoff + 1)):
                        pair_func = eps_scale*one_half*(sympy.tanh(400*(r0_wca_nm - r12_sym)) + 1)*(4*((sigma_ply_nm/r12_sym)**12 - (sigma_ply_nm/r12_sym)**6) + 1)
                        d_pair_func = -pair_func.diff(rij_args[n])
                        dU_pair.append(sympy.lambdify(rij_args, d_pair_func, modules="numpy"))
        dU_funcs.append(dU_pair)
        dU_idxs += dpair_idxs
        dU_d_arg += dU_pair_d_arg
        dU_dxi += dU_pair_dxi
        dU_ck += dU_pair_ck


    # create spline basis functions
    # for each pair
    #    for each spline
    #        lambdify derivative of spline
    #        assign pair to derivative

    n_gauss = 10
    rmin = 3
    rmax = 10 
    gauss_r0 = [ sympy.Rational(rmin + i, 10) for i in range(rmax) ]
    gauss_w = sympy.Rational(1, 10)


    if non_bond_gaussians:
        bond_cutoff = 3
        dU_gauss = []
        dU_gauss_dxi = []
        dU_gauss_ck = []
        dU_gauss_d_arg = []
        dgauss_idxs = []

        # add a gaussian well
        for m in range(len(gauss_r0)):
            dU_m = []
            for i in range(n_beads - bond_cutoff - 1):
                idxs1 = np.arange(3) + i*3
                for j in range(i + bond_cutoff + 1, n_beads):
                    idxs2 = np.arange(3) + j*3
                    xi_idxs = np.concatenate([idxs1, idxs2])

                    # loop over basis functions
                    for n in range(len(rij_args)):
                        dgauss_idxs.append(xi_idxs)
                        dU_gauss_dxi.append(xi_idxs[n])

                        # add 
                        dU_gauss_ck.append(len(dU_funcs) + m)
                        dU_gauss_d_arg.append(n)
                        if (i == 0) and (j == (bond_cutoff + 1)):
                            gauss_func = -gauss_scale*sympy.exp(-one_half*((r12_sym - gauss_r0[m])/gauss_w)**2)

                            d_gauss_func = -gauss_func.diff(rij_args[n])
                            dU_m.append(sympy.lambdify(rij_args, d_gauss_func, modules="numpy"))
            dU_gauss.append(dU_m)

        dU_funcs.extend(dU_gauss)
        dU_idxs += dgauss_idxs
        dU_d_arg += dU_gauss_d_arg
        dU_dxi += dU_gauss_dxi
        dU_ck += dU_gauss_ck

    return dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors

def many_body_function(n_beads, gaussians=True):
    """ """

    # 
    gauss_func = -gauss_scale*sympy.exp(-one_half*((r12_sym - gauss_r0[m])/gauss_w)**2)

    max_n_args = 3*n_beads
    xyz_sym = []
    for i in range(max_n_args/3):
        x_i = sympy.symbols('x' + str(i + 1))
        y_i = sympy.symbols('y' + str(i + 1))
        z_i = sympy.symbols('z' + str(i + 1))
        xyz_sym.append([x_i, y_i, z_i])

    x1, y1, z1 = xyz_sym[0]
    x2, y2, z2 = xyz_sym[1]
    x3, y3, z3 = xyz_sym[2]

    #r0 = 0.38
    #w = 0.1
    one_half = sympy.Rational(1,2)
    r0 = sympy.Rational(38, 100)
    w = sympy.Rational(1, 10)

    # local density
    for i in range(n_beads):
        xi, yi, zi = xyz_sym[i]
        for j in range(n_beads):
            xj, yj, zj = xyz_sym[j]

            #rij_sym = sympy.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
            #rho_ij = one_half*(sympy.tanh(sympy.Rational(r0 - rij_sym, w)) + 1)

    r12_sym = sympy.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r23_sym = sympy.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
    r13_sym = sympy.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)

    rij_args = (x1, y1, z1, x2, y2, z2)
    rho_sym = 1 


def one_dimension():

    x = sympy.symbols("x")

    polynomials = [x] + [ x**i for i in range(2, 10) ]
    sinosoids = [sympy.sin(x)] + [sympy.sin(x*i) for i in range(2, 5)] + [sympy.cos(x)] + [sympy.cos(x*i) for i in range(2, 5)]

    exponentials = [sympy.exp(x), sympy.exp(-x)]
    database = polynomials + sinosoids + exponentials

    U_funcs = []
    dU_funcs = []
    for i in range(len(database)):
        U_funcs.append(sympy.lambdify(x, database[i], modules="numpy"))
        dU_funcs.append(sympy.lambdify(x, -database[i].diff(x), modules="numpy"))

    return U_funcs, dU_funcs


def hermite_1D(domain, n_herm=20):

    xdata = np.linspace(domain[0], domain[1], 1000)
    U_funcs = []
    dU_funcs = []
    for i in range(1, n_herm):
        coeff = np.zeros(n_herm)
        coeff[i] = 1
        #y = np.polynomial.hermite.hermval(xdata, coeff)
        H_i = np.polynomial.hermite.Hermite(coeff, domain=domain)
        max_val = H_i(xdata).max()

        coeff[i] = 1./max_val
        H_i = np.polynomial.hermite.Hermite(coeff, domain=domain)
        dH_i = -H_i.deriv(1)
        U_funcs.append(H_i)
        dU_funcs.append(dH_i)
    return U_funcs, dU_funcs

def gaussians_1D(domain):

    xdata = np.linspace(domain[0], domain[1], 1000)
    U_funcs = []
    dU_funcs = []

    # cover interval in different gaussian basis functions

    for i in range(1, n_herm):
        coeff = np.zeros(n_herm)
        coeff[i] = 1
        #y = np.polynomial.hermite.hermval(xdata, coeff)
        H_i = np.polynomial.hermite.Hermite(coeff, domain=domain)
        max_val = H_i(xdata).max()

        coeff[i] = 1./max_val
        H_i = np.polynomial.hermite.Hermite(coeff, domain=domain)
        dH_i = -H_i.deriv(1)
        U_funcs.append(H_i)
        dU_funcs.append(dH_i)
    return U_funcs, dU_funcs

def Bsplines_1D(domain, n_knots=100, knots=None, k=3):

    # cover interval in spline basis
    if knots is None:
        pad_left = np.array(3*[domain[0]])
        pad_right = np.array(3*[domain[1]])
        knots = np.concatenate((pad_left, np.linspace(domain[0], domain[1], n_knots), pad_right))
    else:
        pad_left = np.array(3*[knots[0]])
        pad_right = np.array(3*[knots[-1]])
        knots = np.concatenate((pad_left, knots, pad_right))

    U_funcs = []
    dU_funcs = []
    for i in range(len(knots) - k - 1):
        coeff = np.zeros(len(knots))
        coeff[i] = 1.
        B_i = scipy.interpolate.BSpline(knots, coeff, k)
        negB_i = scipy.interpolate.BSpline(knots, -coeff, k)
        dB_i = negB_i.derivative(1)
        U_funcs.append(B_i)
        dU_funcs.append(dB_i)

    return U_funcs, dU_funcs

