import time
import numpy as np
import sympy

import scipy.interpolate

import simtk.unit as unit

import simulation.openmm as sop


class CoarseGrainModel(object):
    def __init__(self, n_atoms, n_dim=3):
        """Potential energy terms for polymer
        
        Assigns potential forms to sets of participating coordinates. Includes
        derivatives of each assigned interaction with respect to each
        participating coordinate.

        """
        self.n_atoms = n_atoms
        self.n_dim = n_dim
        self.n_dof = n_dim*n_atoms

        # the total potential has two terms U = [U_0, U_1]
        # U_0 is the fixed term of the potential
        # U_1 is the parametric term of the potential
        self.U_sym = [[],[]]             # functional forms, symbolic
        self.U_funcs = [[],[]]           # functional forms, lambdified
        self.dU_funcs = [[],[]]          # derivative of each form wrt each of its arguments, lambdified

        self.U_scale_factors = [[],[]]   # scale factor of each form
        self.U_coord_idxs = [[],[]]      # coordinate indices assigned to form

        self._define_symbolic_variables()

    def _define_symbolic_variables(self):
        """Define symbolic variables"""

        self.one_half = sympy.Rational(1,2)

        self.xyz_sym = []
        for i in range(self.n_atoms):
            x_i = sympy.symbols('x' + str(i + 1))
            y_i = sympy.symbols('y' + str(i + 1))
            z_i = sympy.symbols('z' + str(i + 1))
            self.xyz_sym.append([x_i, y_i, z_i])

        x1, y1, z1 = self.xyz_sym[0]
        x2, y2, z2 = self.xyz_sym[1]

        # pairwise distance variables
        self.r12_sym = sympy.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # pairwise distance arguments
        self.rij_args = (x1, y1, z1, x2, y2, z2)

        if self.n_atoms > 2:
            x3, y3, z3 = self.xyz_sym[2]
            self.r23_sym = sympy.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
            self.r13_sym = sympy.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)

            # bond angle variable and arguments
            self.theta_ijk_sym = sympy.acos((self.r12_sym**2 + self.r23_sym**2 - self.r13_sym**2)/(2*self.r12_sym*self.r23_sym))
            self.theta_ijk_args = (x1, y1, z1, x2, y2, z2, x3, y3, z3)

        if self.n_atoms > 3:
            # From wikipedia
            # Consider four consequentive atoms bonded as: 1-2-3-4
            # Let
            #  A = r12
            #  B = r23
            #  C = r34

            # and n1 (n2) be the vector normal to plane formed by atoms 1-2-3 (2-3-4)
            # n1 = (A x B)/(|A||B|)
            # n2 = (B x C)/(|B||C|)

            # Then the dihedral is angle between n1 n2
            # cos(phi) = n1*n2

            x3, y3, z3 = self.xyz_sym[2]
            x4, y4, z4 = self.xyz_sym[3]

            normA = sympy.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            normB = sympy.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
            normC = sympy.sqrt((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2)

            normA2 = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
            normB2 = (x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2

            n1x = (y2 - y1)*(z3 - z2) - (z2 - z1)*(y3 - y2)
            n1y = (x2 - x1)*(z3 - z2) - (z2 - z1)*(x3 - x2)
            n1z = (x2 - x1)*(y3 - y2) - (y2 - y1)*(x3 - x2)

            n2x = (y3 - y2)*(z4 - z3) - (z3 - z2)*(y4 - y3)
            n2y = (x3 - x2)*(z4 - z3) - (z3 - z2)*(x4 - x3)
            n2z = (x3 - x2)*(y4 - y3) - (y3 - y2)*(x4 - x3)

            x = n1x*n2x + n1y*n2y + n1z*n2z

            m1x = n1y*(z3 - z2) - n1z*(y3 - y2) 
            m1y = n1x*(z3 - z2) - n1z*(x3 - x2)
            m1z = n1x*(y3 - y2) - n1y*(x3 - x2)

            y = (m1x*n2x + m1y*n2y + m1z*n2z)/normB

            # MDTraj way
            p1 = (n2x*(x2 - x1) + n2y*(y2 - y1) + n2z*(z2 - z1))*normA
            p2 = x

            # TODO: Match MDtraj output
            self.phi_ijkl_sym = sympy.acos(sympy.Abs(x)/(normA*normB2*normC))      # wikipedia
            self.phi_ijkl_sym_mdtraj = sympy.atan2(p1, p2)                      # MDtraj
            #self.phi_ijkl_sym2 = sympy.atan2(y, x)                # stackoverflow

            self.phi_ijkl_args = (x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        # TODO: many-body variables?

    def _add_assignments(self, fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs,
            temp_dU_funcs):

        fixd = int(not fixed) # 0 for fixed, 1 for free
        self.U_sym[fixd].append(U_sym)
        self.U_funcs[fixd].append(U_lamb)
        self.U_scale_factors[fixd].append(scale_factor)
        self.U_coord_idxs[fixd].append(temp_U_coord_idxs)
        self.dU_funcs[fixd].append(temp_dU_funcs)

    def calculate_potential_terms(self, traj):

        xyz_flat = np.reshape(traj.xyz, (traj.n_frames, self.n_dof))
        U_terms = []
        for z in [0,1]:
            temp_U_terms = []
            for i in range(len(self.U_funcs[z])):
                U_i_func = self.U_funcs[z][i]
                U_i_tot = np.zeros(traj.n_frames, float)
                for n in range(len(self.U_coord_idxs[z][i])):
                    xi_idxs = self.U_coord_idxs[z][i][n]
                    U_i_tot += U_i_func(*xyz_flat[:,xi_idxs].T)

                temp_U_terms.append(U_i_tot)
            U_terms.append(temp_U_terms)
        return U_terms

    def calculate_fixed_forces(self, traj, s_frames=0, G=None):
        """Calculate total force due to each parameter along each dimension
        
        Parameters
        ----------
        traj : mdtraj.Trajectory
        
        s_frames : int
            Number of frames to . s_frames > 0 when using the Kramers-Moyal
            relation.
            
        G : None or np.ndarray
            If G is None then new matrix is created otherwise results are added
            to previous calculation.
        """

        xyz_flat = np.reshape(traj.xyz, (traj.n_frames, self.n_dof))
        n_rows = (traj.n_frames - s_frames)*self.n_dof
        n_terms = len(self.U_funcs[0])

        if G is None:
            G = np.zeros(n_rows, float)

        for i in range(n_terms):
            # coordinates assigned fixed potential term i
            coord_idxs = self.U_coord_idxs[0][i]

            for j in range(len(self.dU_funcs[0][i])):
                # derivative wrt argument j
                d_func = self.dU_funcs[0][i][j]
                
                for n in range(len(coord_idxs)):
                    # coordinates assigned to this derivative
                    if s_frames == 0:
                        deriv = -d_func(*xyz_flat[:,coord_idxs[n]].T)
                    else:
                        deriv = -d_func(*xyz_flat[:,coord_idxs[n]].T)[:-s_frames]

                    # derivative is wrt to coordinate index dxi
                    dxi = coord_idxs[n][j]
                    xi_ravel_idxs = np.arange(dxi, n_rows, self.n_dof)
                    G[xi_ravel_idxs] += deriv.ravel()
        return G

    def calculate_parametric_forces(self, traj, s_frames=0, G=None):
        """Calculate total force due to each parameter along each dimension
        
        Parameters
        ----------
        traj : mdtraj.Trajectory
        
        s_frames : int
            Number of frames to . s_frames > 0 when using the Kramers-Moyal
            relation.
            
        G : None or np.ndarray
            If G is None then new matrix is created otherwise results are added
            to previous calculation.

        Returns
        -------
        G : np.ndarray (T, P)
            Matrix of 
        """

        xyz_flat = np.reshape(traj.xyz, (traj.n_frames, self.n_dof))
        n_rows = (traj.n_frames - s_frames)*self.n_dof
        n_params = len(self.U_funcs[1])

        if G is None:
            G = np.zeros((n_rows, n_params), float)

        for i in range(n_params): 
            # parameter i corresponds to functional form i
            # coords assigned to functional form i
            coord_idxs = self.U_coord_idxs[1][i]    # I_r

            for j in range(len(self.dU_funcs[1][i])):
                # derivative wrt argument j
                d_func = self.dU_funcs[1][i][j]
                
                for n in range(len(coord_idxs)):
                    # coordinates assigned to this derivative
                    if s_frames == 0:
                        deriv = -d_func(*xyz_flat[:,coord_idxs[n]].T)
                    else:
                        deriv = -d_func(*xyz_flat[:,coord_idxs[n]].T)[:-s_frames]

                    # derivative is wrt to coordinate index dxi
                    dxi = coord_idxs[n][j]
                    xi_ravel_idxs = np.arange(dxi, n_rows, self.n_dof)

                    # force on each coordinate is separated by associated
                    # parameter
                    G[xi_ravel_idxs, i] += deriv.ravel()
        return G

class PolymerModel(CoarseGrainModel):

    def __init__(self, n_atoms):
        """Potential energy terms for polymer
        
        Assigns potential forms to sets of participating coordinates. Includes
        derivatives of each assigned interaction with respect to each
        participating coordinate.

        """
        CoarseGrainModel.__init__(self, n_atoms)

    def _generate_pairwise_idxs(self, bond_cutoff=3):
        coord_idxs = []
        for i in range(self.n_atoms - bond_cutoff):
            idxs1 = np.arange(3) + i*3
            for j in range(i + bond_cutoff, self.n_atoms):
                # assign coordinates to interaction/derivative
                idxs2 = np.arange(3) + j*3
                xi_idxs = np.concatenate([idxs1, idxs2])
                coord_idxs.append(xi_idxs)
        return coord_idxs

    def _generate_bonded_idxs(self, n_args):

        coord_idxs = []
        for i in range(self.n_atoms - (n_args/self.n_dim) + 1):
            # assign coordinates to interaction/derivative
            xi_idxs = np.arange(n_args) + i*3
            coord_idxs.append(xi_idxs)
        return coord_idxs


    def _assign_harmonic_bonds(self, r0_nm, scale_factor=1, fixed=False):
        """Assign harmonic bond interactions
        
        Parameters
        ----------
        r0_nm : float
            Center of harmonic well in nanometers

        scale_factor : float
            Factor to scale function by. If fixed=True then this is coefficient. 

        fixed : bool
            If True then terms are treated as a fixed part of the potential.
        """

        U_sym = scale_factor*self.one_half*((self.r12_sym - r0_nm)**2)
        U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")
        n_args = len(self.rij_args)

        temp_U_coord_idxs = []
        # list is the same for each n. no need to duplicate
        for i in range(self.n_atoms - 1):
            # assign coordinates to interaction/derivative
            xi_idxs = np.arange(n_args) + i*3
            temp_U_coord_idxs.append(xi_idxs)   

        temp_dU_funcs = []
        for n in range(len(self.rij_args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(self.rij_args[n])
            temp_dU_funcs.append(sympy.lambdify(self.rij_args, dU_sym, modules="numpy"))

        self._add_assignments(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def _assign_harmonic_angles(self, theta0_rad, scale_factor=1, fixed=False):
        """Assign harmonic angle interactions
        
        Parameters
        ----------
        theta0_rad : float
            Center of harmonic well in radians

        scale_factor : float
            Factor to scale function by. If fixed=True then this is coefficient. 

        fixed : bool
            If True then terms are treated as a fixed part of the potential.
        """

        assert self.n_atoms > 2, "Not enough number of atoms to have angle interactions"
        
        U_sym = scale_factor*self.one_half*(self.theta_ijk_sym - theta0_rad)**2
        U_lamb = sympy.lambdify(self.theta_ijk_args, U_sym, modules="numpy")
        n_args = len(self.theta_ijk_args)

        temp_U_coord_idxs = []
        for i in range(self.n_atoms - 2):
            # assign coordinates to interaction/derivative 
            xi_idxs = np.arange(n_args) + i*3
            temp_U_coord_idxs.append(xi_idxs)

        temp_dU_funcs = []
        for n in range(len(self.theta_ijk_args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(self.theta_ijk_args[n])
            temp_dU_funcs.append(sympy.lambdify(self.theta_ijk_args, dU_sym, modules="numpy"))

        self._add_assignments(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def _assign_inverse_r12(self, sigma_nm, scale_factor=1, fixed=False, bond_cutoff=3):

        U_sym = scale_factor*(sigma_nm**12)*(self.r12_sym**(-12))
        U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

        temp_U_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)

        temp_dU_funcs = []
        for n in range(len(self.rij_args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(self.rij_args[n])
            temp_dU_funcs.append(sympy.lambdify(self.rij_args, dU_sym, modules="numpy"))

        self._add_assignments(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def _assign_LJ6(self, sigma_nm, scale_factor=1, fixed=False, bond_cutoff=3):

        U_sym = 4*scale_factor*((sigma_nm/self.r12_sym)**(12) - (sigma_nm/self.r12_sym)**6)
        U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

        #temp_U_coord_idxs = self._generate_14_pairwise_idxs(bond_cutoff=bond_cutoff)
        temp_U_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)

        temp_dU_funcs = []
        for n in range(len(self.rij_args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(self.rij_args[n])
            temp_dU_funcs.append(sympy.lambdify(self.rij_args, dU_sym, modules="numpy"))

        self._add_assignments(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def _assign_pairwise_gaussians(self, r0_nm, w_nm, scaling=1, fixed=False, bond_cutoff=3):
        """Assign a Gaussian well a each position"""
        
        for m in range(len(r0_nm)):
            # gaussian well at position r0_nm[m]
            U_sym = -scaling_factor*sympy.exp(-self.one_half*((self.r12_sym - r0_nm[m])/w_nm[m])**2)
            U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

            temp_U_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)

            temp_dU_funcs = []
            for n in range(len(self.rij_args)):
                # take derivative wrt argument n
                dU_sym = U_sym.diff(self.rij_args[n])
                temp_dU_funcs.append(sympy.lambdify(self.rij_args, dU_sym, modules="numpy"))

            self._add_assignments(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

def polymer_library(n_beads, bonds=True, angles=True, non_bond_wca=True, non_bond_gaussians=True):
    # Soon to be deprecated oct 2018

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
        bond_sym = kb_scale*one_half*(r12_sym - r0_nm)**2 # scaled

        U_bond = sympy.lambdify(rij_args, bond_sym, modules="numpy")

        bond_arg_idxs = []
        dU_bond = []
        dU_bond_dxi = []
        #dU_bond_ck = []
        dU_bond_d_arg = []
        for i in range(n_beads - 1):
            xi_idxs = np.arange(6) + i*3
            for n in range(len(rij_args)):
                bond_arg_idxs.append(xi_idxs)
                dU_bond_d_arg.append(n)
                dU_bond_dxi.append(xi_idxs[n])
                #dU_bond_ck.append(len(dU_funcs))
                if i == 0:
                    # take derivative wrt argument n
                    d_bond_sym = -bond_sym.diff(rij_args[n])
                    dU_bond.append(sympy.lambdify(rij_args, d_bond_sym, modules="numpy"))

        ck_bond = len(dU_bond_d_arg)*[len(dU_funcs)]    # parameter for each 
        dU_funcs.append(dU_bond)
        dU_idxs += bond_arg_idxs
        dU_d_arg += dU_bond_d_arg
        dU_dxi += dU_bond_dxi
        #dU_ck += dU_bond_ck
        dU_ck += ck_bond

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

def Bsplines_1D(domain, n_knots=100, knots=None, k=3, second_deriv=False):

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

    if second_deriv:
        d2U_funcs = []
        for i in range(len(knots) - k - 1):
            coeff = np.zeros(len(knots))
            coeff[i] = 1.
            B_i = scipy.interpolate.BSpline(knots, coeff, k)
            d2B_i = B_i.derivative(2)
            d2U_funcs.append(d2B_i)
        return U_funcs, dU_funcs, d2U_funcs
    else:
        return U_funcs, dU_funcs

