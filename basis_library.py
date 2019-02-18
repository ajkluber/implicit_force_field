from __future__ import print_function, absolute_import
import time
import sys
import numpy as np
import sympy

import scipy.interpolate
from scipy.stats import binned_statistic as bin1d

import mdtraj as md
import simtk.unit as unit

import simulation.openmm as sop


class FunctionLibrary(object):
    def __init__(self, n_atoms, beta, n_dim=3, using_cv=False, using_D2=False, periodic_box_dims=[]):
        """Potential energy terms for polymer
        
        Assigns potential forms to sets of participating coordinates. Includes
        derivatives of each assigned interaction with respect to each
        participating coordinate.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system.

        n_dims : int, optional
            Dimensionality.

        periodic_box_dims : list, (n_dims)
            List of periodic box dimensions in the x,y,z directions. 
            Input should be [L_x, L_y, L_z] where L_i is the length of box
            in i direction.
        """
        self.n_atoms = n_atoms
        self.n_dim = n_dim
        self.n_dof = n_dim*n_atoms
        self.box_dims = periodic_box_dims
        self.beta = beta

        self.using_D2 = using_D2
        self.using_cv = using_cv
        self.using_U0 = False
        self.constant_a_coeff = True
        self.fixed_a_coeff = False
        self.cv_defined = False

        # the total potential has two terms U = [U_0, U_1]
        # U_0 is the fixed term of the potential
        # U_1 is the parametric term of the potential
        self.U_sym = [[],[]]             # functional forms, symbolic
        self.U_funcs = [[],[]]           # functional forms, lambdified
        self.dU_funcs = [[],[]]          # derivative of each form wrt each of its arguments, lambdified

        self.U_scale_factors = [[],[]]   # scale factor of each form
        self.U_coord_idxs = [[],[]]      # coordinate indices assigned to form

        # Cartesian coordinate test functions
        self.f_sym = []             # functional forms, symbolic
        self.f_funcs = []           # functional forms, lambdified
        self.f_coord_idxs = []      # coordinate indices assigned to form
        self.df_funcs = []          # first derivative of each form wrt each of its arguments, lambdified
        self.d2f_funcs = []         # second derivative of each form wrt each of its arguments, lambdified

        # Collective variable (CV) potential functions
        self.cv_U_sym = []             # functional forms, symbolic
        self.cv_U_funcs = []           # functional forms, lambdified
        self.cv_dU_funcs = []          # first derivative of each form wrt each of its arguments, lambdified

        # Collective variable (CV) noise functions
        self.cv_a_sym = []             # functional forms, symbolic
        self.cv_a_funcs = []           # functional forms, lambdified
        self.cv_da_funcs = []          # first derivative of each form wrt each of its arguments, lambdified

        # Collective variable (CV) test functions
        self.cv_f_sym = []             # functional forms, symbolic
        self.cv_f_funcs = []           # functional forms, lambdified
        self.cv_df_funcs = []          # first derivative of each form wrt each of its arguments, lambdified
        self.cv_d2f_funcs = []         # second derivative of each form wrt each of its arguments, lambdified

        # Collective variables are linear combination of features (chi)
        self.chi_sym = []
        self.chi_funcs = []
        self.chi_coeff = [] 
        self.chi_mean = [] 
        self.chi_coord_idxs = []
        self.dchi_funcs = []
        self.d2chi_funcs = []

        self._define_symbolic_variables()

    @property
    def n_params(self):
        if hasattr(self, "using_cv") and self.using_cv:
            return len(self.cv_U_funcs)
        else:
            return len(self.U_funcs[1])

    @property
    def n_test_funcs(self):
        if self.using_cv:
            return len(self.cv_f_funcs)
        else:
            return int(np.sum([ len(self.f_coord_idxs[i]) for i in range(len(self.f_funcs)) ]))

    @property
    def n_cv_dim(self):
        return self.chi_coeff[0].shape[1]

    @property
    def n_feature_types(self):
        return len(self.chi_coeff)

    @property
    def n_features(self):
        return np.sum([ x.shape[0] for x in self.chi_coeff ])

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
        self.x_sym = sympy.symbols("x")

        if self.n_atoms > 1:
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

    def _add_potential_term(self, fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs,
            temp_dU_funcs):

        if fixed:
            self.using_U0 = True

        fixd = int(not fixed) # 0 for fixed, 1 for free
        self.U_sym[fixd].append(U_sym)
        self.U_funcs[fixd].append(U_lamb)
        self.U_scale_factors[fixd].append(scale_factor)
        self.U_coord_idxs[fixd].append(temp_U_coord_idxs)
        self.dU_funcs[fixd].append(temp_dU_funcs)

    def _add_test_functions(self, f_sym, f_lamb, temp_f_coord_idxs, temp_df_funcs, temp_d2f_funcs):

        self.f_sym.append(f_sym)
        self.f_funcs.append(f_lamb)
        self.f_coord_idxs.append(temp_f_coord_idxs)
        self.df_funcs.append(temp_df_funcs)
        self.d2f_funcs.append(temp_d2f_funcs)

    def calculate_potential_terms(self, xyz_traj):

        U_terms = []
        for z in [0,1]:
            temp_U_terms = []
            for i in range(len(self.U_funcs[z])):
                U_i_func = self.U_funcs[z][i]
                U_i_tot = np.zeros(xyz_traj.shape[0], float)
                for n in range(len(self.U_coord_idxs[z][i])):
                    xi_idxs = self.U_coord_idxs[z][i][n]
                    U_i_tot += U_i_func(*xyz_traj[:,xi_idxs].T)

                temp_U_terms.append(U_i_tot)
            U_terms.append(temp_U_terms)
        return U_terms

    def calculate_fixed_forces(self, xyz_traj, s_frames=0, G=None):
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

        n_rows = (xyz_traj.shape[0] - s_frames)*self.n_dof
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
                        deriv = -d_func(*xyz_traj[:,coord_idxs[n]].T)
                    else:
                        deriv = -d_func(*xyz_traj[:,coord_idxs[n]].T)[:-s_frames]

                    # derivative is wrt to coordinate index dxi
                    dxi = coord_idxs[n][j]
                    xi_ravel_idxs = np.arange(dxi, n_rows, self.n_dof)
                    G[xi_ravel_idxs] += deriv.ravel()
        return G

    def calculate_parametric_forces(self, xyz_traj, s_frames=0, G=None):
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

        n_rows = (xyz_traj.shape[0] - s_frames)*self.n_dof

        if G is None:
            G = np.zeros((n_rows, self.n_params), float)

        for i in range(self.n_params): 
            # parameter i corresponds to functional form i
            # coords assigned to functional form i
            coord_idxs = self.U_coord_idxs[1][i]    # I_r

            for j in range(len(self.dU_funcs[1][i])):
                # derivative wrt argument j
                d_func = self.dU_funcs[1][i][j]
                
                for n in range(len(coord_idxs)):
                    # coordinates assigned to this derivative
                    if s_frames == 0:
                        deriv = -d_func(*xyz_traj[:,coord_idxs[n]].T)
                    else:
                        deriv = -d_func(*xyz_traj[:,coord_idxs[n]].T)[:-s_frames]

                    # derivative is wrt to coordinate index dxi
                    dxi = coord_idxs[n][j]
                    xi_ravel_idxs = np.arange(dxi, n_rows, self.n_dof)

                    # force on each coordinate is separated by associated
                    # parameter
                    G[xi_ravel_idxs, i] += deriv.ravel()
        return G


class OneDimensionalModel(FunctionLibrary):
    # TODO: Simpler support for low-dimension systems

    def __init__(self, n_atoms, periodic_box_dims=[]):
        """One-dimensional model
        
        Potential forms, test functions, and their derivatives.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system.

        periodic_box_dims : list, [x_L, x_R]
            Positions of the left and right boundaries, x_L and x_R,
            respectively.
        """

        self.n_atoms = n_atoms
        self.n_dim = 1
        self.n_dof = n_atoms
        self.box_dims = periodic_box_dims

        # symbolic variable
        self.x_sym = sympy.symbols("x")
        self.one_half = sympy.Rational(1,2)

        # the drift is expanded in basis functions
        # has two terms b = [b_0, b_1]
        # b_0 is a fixed term
        # b_1 is a parametric term
        self.b_sym = [[],[]]             # functional forms, symbolic
        self.b_funcs = [[],[]]           # functional forms, lambdified
        self.d2b_funcs = [[],[]]           # functional forms, lambdified
        self.b_scale_factors = [[],[]]   # scale factor of each form

        # the noise is expanded in basis functions 
        self.a_sym = [[],[]]             # functional forms, symbolic
        self.a_funcs = [[],[]]           # functional forms, lambdified
        self.d2a_funcs = [[],[]]           # functional forms, lambdified
        self.a_scale_factors = [[],[]]   # scale factor of each form

        # test functions
        self.f_sym = []             # functional forms, symbolic
        self.f_funcs = []           # functional forms, lambdified
        self.df_funcs = []          # first derivative of each form wrt each of its arguments, lambdified
        self.d2f_funcs = []         # second derivative of each form wrt each of its arguments, lambdified

    def add_Gaussian_drift_basis(self, r0, w, scale_factor=1, fixed=False):
        """Assign a Gaussian well a each position"""
        
        fixd = int(not fixed) # 0 for fixed, 1 for free
        for m in range(len(r0)):
            # gaussian well at position r0_nm[m]
            b_sym = scale_factor*sympy.exp(-self.one_half*((self.x_sym - r0[m])/w[m])**2)
            b_lamb = sympy.lambdify(self.x_sym, b_sym, modules="numpy")
            d2b_lamb = sympy.lambdify(self.x_sym, b_sym.diff(self.x_sym, 2), modules="numpy")
            self.b_sym[fixd].append(b_sym)
            self.b_funcs[fixd].append(b_lamb)
            self.d2b_funcs[fixd].append(d2b_lamb)
            self.b_scale_factors[fixd].append(scale_factor)

    def add_Gaussian_noise_basis(self, r0, w, scale_factor=1, fixed=False):
        """Assign a Gaussian well a each position"""
        
        fixd = int(not fixed) # 0 for fixed, 1 for free
        for m in range(len(r0)):
            # gaussian well at position r0_nm[m]
            a_sym = scale_factor*sympy.exp(-self.one_half*((self.x_sym - r0[m])/w[m])**2)
            a_lamb = sympy.lambdify(self.x_sym, a_sym, modules="numpy")
            d2a_lamb = sympy.lambdify(self.x_sym, a_sym.diff(self.x_sym, 2), modules="numpy")
            self.a_sym[fixd].append(a_sym)
            self.a_funcs[fixd].append(a_lamb)
            self.d2a_funcs[fixd].append(d2a_lamb)
            self.a_scale_factors[fixd].append(scale_factor)

    def add_Gaussian_test_functions(self, r0, w, scale_factor=1, fixed=False):
        """Assign a Gaussian well a each position"""
        
        for m in range(len(r0)):
            # gaussian well at position r0_nm[m]
            f_sym = scale_factor*sympy.exp(-self.one_half*((self.x_sym - r0[m])/w[m])**2)
            f_lamb = sympy.lambdify(self.x_sym, f_sym, modules="numpy")
            df_func = sympy.lambdify(self.x_sym, f_sym.diff(self.x_sym), modules="numpy") 
            d2f_func = sympy.lambdify(self.x_sym, f_sym.diff(self.x_sym, 2), modules="numpy") 

            self.f_sym.append(f_sym)
            self.f_funcs.append(f_lamb)
            self.df_funcs.append(df_func)
            self.d2f_funcs.append(d2f_func)

    #########################################################
    # EVALUATE DRIFT AND NOISE BASIS FUNCTIONS ON TRAJECTORY
    #########################################################
    def evaluate_fixed_drift(self, x_traj):
        """Fixed term of the drift"""
        drift_0 = np.zeros(x_traj.shape[0], float)
        for i in range(len(self.b_funcs[0])):
            drift_0 += self.b_funcs[0][i](x_traj) 
        return drift_0

    def evaluate_parametric_drift(self, x_traj):
        """Parametric term of the drift"""
        drift_1 = np.zeros((x_traj.shape[0], len(self.b_funcs[1])), float)
        for i in range(len(self.b_funcs[1])):
            drift_1[:,i] = self.b_funcs[1][i](x_traj) 
        return drift_1

    def evaluate_fixed_noise(self, x_traj):
        """Fixed term of the noise"""
        noise_0 = np.zeros(x_traj.shape[0], float)
        for i in range(len(self.a_funcs[0])):
            noise_0 += self.a_funcs[0][i](x_traj) 
        return noise_0

    def evaluate_parametric_noise(self, x_traj):
        """Parametric term of the noise"""
        noise_1 = np.zeros((x_traj.shape[0], len(self.a_funcs[1])), float)
        for i in range(len(self.a_funcs[1])):
            noise_1[:,i] = self.a_funcs[1][i](x_traj) 
        return noise_1

    def evaluate_D2_matrix(self, x_traj):
        """Second derivatives"""

        n_b = len(self.d2b_funcs[1])
        N = n_b + len(self.d2a_funcs[1])
        D2 = np.zeros((N, N), float)

        for i in range(len(self.d2b_funcs[1])):
            d2b_i = self.d2b_funcs[1][i](x_traj)
            for j in range(len(self.d2b_funcs[1])):
                d2b_j = self.d2b_funcs[1][j](x_traj)
                D2[i,j] = np.sum(d2b_i*d2b_j)

        for i in range(len(self.d2a_funcs[1])):
            d2a_i = self.d2a_funcs[1][i](x_traj)
            for j in range(len(self.d2a_funcs[1])):
                d2a_j = self.d2a_funcs[1][j](x_traj)
                D2[n_b + i,n_b + j] = np.sum(d2a_i*d2a_j)

        return D2 


    #########################################################
    # EVALUATE TEST FUNCTIONS ON TRAJECTORY
    #########################################################
    def test_functions(self, x_traj):
        """Test functions"""

        test_f = np.zeros((x_traj.shape[0], len(self.f_funcs)), float)
        for j in range(len(self.f_funcs)):
            # test function form j
            f_j_func = self.f_funcs[j]
            test_f[:, j] = f_j_func(x_traj)
        return test_f

    def test_funcs_gradient_and_laplacian(self, x_traj):
        """Gradient of test functions"""

        grad_f = np.zeros((x_traj.shape[0], len(self.f_funcs)), float)
        for j in range(len(self.f_funcs)):
            # test function form j
            df_func = self.df_funcs[j]
            grad_f[:, j] = df_func(x_traj)

        Lap_f = np.zeros((x_traj.shape[0], len(self.f_funcs)), float)
        for j in range(len(self.f_funcs)):
            # test function form j
            d2f_func = self.d2f_funcs[j]
            Lap_f[:, j] = d2f_func(x_traj)

        return grad_f, Lap_f

class PolymerModel(FunctionLibrary):

    def __init__(self, n_atoms, beta, bond_cutoff=4, using_cv=False,
            using_D2=False, constant_a_coeff=True, a_coeff=None):
        """Potential energy terms for polymer
        
        Assigns potential forms to sets of participating coordinates. Includes
        derivatives of each assigned interaction with respect to each
        participating coordinate.

        """
        FunctionLibrary.__init__(self, n_atoms, beta, using_cv=using_cv, using_D2=using_D2)
        self.bond_cutoff = bond_cutoff

        if a_coeff is None:
            self.fixed_a_coeff = False
            self.a_coeff = None
        else:
            if not constant_a_coeff:
                raise ValueError("To set a fixed diffusion coefficient, the model must be created with constant_a_coeff=True.")
            self.fixed_a_coeff = True
            self.a_coeff = a_coeff

    ##########################################################
    # DEFINE COLLECTIVE VARIABLE
    ##########################################################
    def linear_collective_variables(self, feature_types, feature_atm_idxs, feature_coeff, feature_mean):
        """Define collective variables (cv) in terms of feature functions
        symbolically and take derivate of features wrt each Cartesian coord.
        
        Collective variables are allowed to be linear combination of features
        which are functions of cartesian coordinates.

        You should use, for example, the result of Time Independent Component
        Analysis (TICA) as collective variables.

        Parameters
        ----------
        feature_types : list (n_features)
            Feature types used to define the collective variable.

        feature_atm_idxs : list of arrays
            Each element is a list of atom indices that participate in the
            corresponding feature.

        feature_coeff : list of arrays
            Each element is an array of coefficients that multiple the
            corresponding feature.

        feature_mean : np.ndarray
            Mean value of each feature.

        """

        #TODO: Add more types of features (e.g., angle, dihedral, etc.)

        n_cv_dim = feature_coeff.shape[1]
        self.cv_sym = [ sympy.symbols("psi" + str(i + 1)) for i in range(n_cv_dim) ]
        self.cv_args = tuple(self.cv_sym)

        available_feature_types = {"dist":self.r12_sym, "invdist":1/self.r12_sym}
        feature_type_args = {"dist":self.rij_args, "invdist":self.rij_args}

        for m in range(len(feature_types)):
            if not feature_types[m] in available_feature_types:
                raise ValueError(feature_types[m] + " not in available features: " + " ".join(available_feature_types.keys()))

        for m in range(len(feature_types)):
            # symbolic feature function and variables
            feat_sym = available_feature_types[feature_types[m]]
            feat_args = feature_type_args[feature_types[m]] 

            # collective variables are linear combination of features
            self.chi_coeff.append(feature_coeff)
            self.chi_mean.append(feature_mean)

            # feature function
            self.chi_sym.append(feat_sym)
            self.chi_funcs.append(sympy.lambdify(feat_args, feat_sym, modules="numpy"))

            # first and second derivative of feature function wrt each cartesian
            # coordinate
            temp_dchi = []
            temp_d2chi = []
            for n in range(len(self.rij_args)):
                d_chi = feat_sym.diff(feat_args[n])
                d2_chi = feat_sym.diff(feat_args[n], 2)
                temp_dchi.append(sympy.lambdify(feat_args, d_chi, modules="numpy"))
                temp_d2chi.append(sympy.lambdify(feat_args, d2_chi, modules="numpy"))
            self.dchi_funcs.append(temp_dchi)
            self.d2chi_funcs.append(temp_d2chi)

            # assign coordinate indices to for feature
            temp_coord_idxs = []
            for i in range(len(feature_atm_idxs)):
                # determine participating coordinate indices from atom indices
                temp_idxs = []
                for z in range(len(feature_atm_idxs[i])):
                    atm_z = feature_atm_idxs[i][z]
                    temp_idxs.append(np.arange(self.n_dim) + atm_z*self.n_dim)
                xi_idxs = np.concatenate(temp_idxs) 
                temp_coord_idxs.append(xi_idxs)
            self.chi_coord_idxs.append(temp_coord_idxs)

        self.cv_defined = True

    ##########################################################
    # POTENTIAL FUNCTIONS
    ##########################################################
    def _generate_pairwise_idxs(self, bond_cutoff=3, sort_by_seq_sep=False):
        if sort_by_seq_sep:
            idxs_by_seq_sep = [ [] for i in range(self.n_atoms - bond_cutoff) ]
            for i in range(self.n_atoms - bond_cutoff):
                idxs1 = np.arange(3) + i*3
                for j in range(i + bond_cutoff, self.n_atoms):
                    # assign coordinates to interaction/derivative
                    idxs2 = np.arange(3) + j*3
                    xi_idxs = np.concatenate([idxs1, idxs2])
                    idxs_by_seq_sep[j - i - bond_cutoff].append(xi_idxs)

            coord_idxs = []
            for i in range(len(idxs_by_seq_sep)):
                coord_idxs.append(idxs_by_seq_sep[i])
        else:
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
        raise NotImplementedError

        coord_idxs = []
        for i in range(self.n_atoms - (n_args/self.n_dim) + 1):
            # assign coordinates to interaction/derivative
            xi_idxs = np.arange(n_args) + i*3
            coord_idxs.append(xi_idxs)
        return coord_idxs

    def harmonic_bond_potentials(self, r0_nm, scale_factor=1, fixed=False):
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

        self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def harmonic_angle_potentials(self, theta0_rad, scale_factor=1, fixed=False):
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

        self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def inverse_r12_potentials(self, sigma_nm, scale_factor=1, fixed=False, bond_cutoff=3):

        U_sym = scale_factor*(sigma_nm**12)*(self.r12_sym**(-12))
        U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

        temp_U_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)

        temp_dU_funcs = []
        for n in range(len(self.rij_args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(self.rij_args[n])
            temp_dU_funcs.append(sympy.lambdify(self.rij_args, dU_sym, modules="numpy"))

        self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def LJ6_potentials(self, sigma_nm, scale_factor=1, fixed=False, bond_cutoff=3):

        U_sym = 4*scale_factor*((sigma_nm/self.r12_sym)**(12) - (sigma_nm/self.r12_sym)**6)
        U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

        #temp_U_coord_idxs = self._generate_14_pairwise_idxs(bond_cutoff=bond_cutoff)
        temp_U_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)

        temp_dU_funcs = []
        for n in range(len(self.rij_args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(self.rij_args[n])
            temp_dU_funcs.append(sympy.lambdify(self.rij_args, dU_sym, modules="numpy"))

        self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def _take_symbolic_derivatives(self, U_sym, args):
        """Take symbolic derivative of U_sym wrt each argument"""
        temp_dU_funcs = []
        for n in range(len(args)):
            # take derivative wrt argument n
            dU_sym = U_sym.diff(args[n])
            temp_dU_funcs.append(sympy.lambdify(args, dU_sym, modules="numpy"))
        return temp_dU_funcs

    def gaussian_pair_potentials(self, r0_nm, w_nm, scale_factor=1,
            fixed=False, bond_cutoff=3, symmetry="shared"):
        """Assign a Gaussian well a each position
        
        Parameters
        ----------
        r0_nm : np.array 
            Centers of Gaussians

        w_nm : np.array 
            Widths of Gaussians

        scale_factor : float
            Prefactor.

        fixed : bool
            Indicates if term is held fixed or parametric.

        bond_cutoff : int
            Pairs with |i - j| < bond_cutoff are excluded from this interaction.

        symmetry : str
            shared = All pairs use the same interaction.
            seq_sep = All pairs with the same |i - j| have same interaction.
            unique = Each pair has its own unique interaction.

            The number of free parameters grows from shared -> seq_sep -> unique.
        """
        
        if symmetry == "shared":
            for m in range(len(r0_nm)):
                # gaussian well at position r0_nm[m]
                U_sym = -scale_factor*sympy.exp(-self.one_half*((self.r12_sym - r0_nm[m])/w_nm[m])**2)
                U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

                temp_U_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)
                temp_dU_funcs = self._take_symbolic_derivatives(U_sym, self.rij_args) 
                self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)
        elif symmetry == "seq_sep":
            coord_idxs_by_seq_sep = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff, sort_by_seq_sep=True)

            for i in range(len(coord_idxs_by_seq_sep)):
                temp_U_coord_idxs = coord_idxs_by_seq_sep[i]

                for m in range(len(r0_nm)):
                    # gaussian well at position r0_nm[m]
                    U_sym = -scale_factor*sympy.exp(-self.one_half*((self.r12_sym - r0_nm[m])/w_nm[m])**2)
                    U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

                    temp_dU_funcs = self._take_symbolic_derivatives(U_sym, self.rij_args) 
                    self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

        elif symmetry == "unique":
            all_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff, sort_by_seq_sep=False)

            for i in range(len(all_coord_idxs)):
                temp_U_coord_idxs = [all_coord_idxs[i]]

                for m in range(len(r0_nm)):
                    # gaussian well at position r0_nm[m]
                    U_sym = -scale_factor*sympy.exp(-self.one_half*((self.r12_sym - r0_nm[m])/w_nm[m])**2)
                    U_lamb = sympy.lambdify(self.rij_args, U_sym, modules="numpy")

                    temp_dU_funcs = self._take_symbolic_derivatives(U_sym, self.rij_args) 
                    self._add_potential_term(fixed, U_sym, U_lamb, scale_factor, temp_U_coord_idxs, temp_dU_funcs)

    def gaussian_cv_potentials(self, cv_r0, cv_w, scale_factors=1):

        if scale_factors == 1:
            scale_factors = np.ones(len(cv_r0))
        else:
            if len(scale_factors) != len(cv_r0):
                raise ValueError("Number of scale factors should match number of centers")

        for n in range(len(cv_r0)):
            # basis function is Gaussian with center r0 and width w
            f_sym = (self.cv_sym[0] - cv_r0[n][0])**2
            for i in range(1, self.n_cv_dim):
                f_sym += (self.cv_sym[i] - cv_r0[n][i])**2
            f_sym = scale_factors[n]*sympy.exp(-self.one_half*f_sym/(cv_w[n]**2))
            f_lamb = sympy.lambdify(self.cv_args, f_sym, modules="numpy")

            self.cv_U_sym.append(f_sym)
            self.cv_U_funcs.append(f_lamb)

            # first and second derivative wrt each arg
            temp_cv_dU_funcs = self._take_symbolic_derivatives(f_sym, self.cv_args) 
            self.cv_dU_funcs.append(temp_cv_dU_funcs)

    def gaussian_cv_noise_functions(self, cv_r0, cv_w):
        
        raise NotImplementedError
        for n in range(len(cv_r0)):
            # basis function is Gaussian with center r0 and width w
            f_sym = (self.cv_sym[0] - cv_r0[n][0])**2
            for i in range(1, self.n_cv_dim):
                f_sym += (self.cv_sym[i] - cv_r0[n][i])**2
            f_sym = sympy.exp(-self.one_half*f_sym/(cv_w[n]**2))
            f_lamb = sympy.lambdify(self.cv_args, f_sym, modules="numpy")

            self.cv_a_sym.append(f_sym)
            self.cv_a_funcs.append(f_lamb)

            # first and second derivative wrt each arg
            temp_cv_da_funcs = []
            for i in range(len(self.cv_args)):
                df_sym = f_sym.diff(self.cv_args[i])
                temp_cv_da_funcs.append(sympy.lambdify(self.cv_args, df_sym, modules="numpy"))

            self.cv_da_funcs.append(temp_cv_da_funcs)

    ##########################################################3
    # TEST FUNCTIONS
    ##########################################################3
    def gaussian_bond_test_funcs(self, r0_nm, w_nm, coeff=1):
        """Assign harmonic bond interactions
        
        Parameters
        ----------
        r0_nm : np.ndarry float
            Center of Gaussians for each bond in nanometers

        w_nm : np.ndarry float
            Widths of Gaussians for each bond in nanometers

        coeff : float or np.ndarray
            Coefficient to multiple each test function
        """

        n_args = len(self.rij_args)
        if coeff == 1:
            coeff = np.ones(len(r0_nm))

        temp_f_coord_idxs = []
        # list is the same for each n. no need to duplicate
        for i in range(self.n_atoms - 1):
            # assign coordinates to interaction/derivative
            xi_idxs = np.arange(n_args) + i*3
            temp_f_coord_idxs.append(xi_idxs)   

        # Add a couple gaussians along each bond
        for n in range(len(r0_nm)):
            # gaussian well at position r0_nm[m]
            f_sym = coeff[n]*sympy.exp(-self.one_half*((self.r12_sym - r0_nm[n])/w_nm[n])**2)
            f_lamb = sympy.lambdify(self.rij_args, f_sym, modules="numpy")

            temp_df_funcs = []
            temp_d2f_funcs = []
            for i in range(len(self.rij_args)):
                # take derivative wrt argument n
                df_sym = f_sym.diff(self.rij_args[i])
                d2f_sym = df_sym.diff(self.rij_args[i])
                temp_df_funcs.append(sympy.lambdify(self.rij_args, df_sym, modules="numpy"))
                temp_d2f_funcs.append(sympy.lambdify(self.rij_args, d2f_sym, modules="numpy"))

            self._add_test_functions(f_sym, f_lamb, temp_f_coord_idxs, temp_df_funcs, temp_d2f_funcs)

    def vonMises_angle_test_funcs(self, theta0_rad, kappa, coeff=1):
        """Assign harmonic angle interactions
        
        Parameters
        ----------
        theta0_rad : np.ndarray, float
            Center of Von Mises function in radians

        kappa : np.ndarray, float
            Steepness of Von Mises function
        
        coeff : float or np.ndarray
            Coefficient to multiple each test function
        
        """

        assert self.n_atoms > 2, "Not enough number of atoms to have angle interactions"
        n_args = len(self.theta_ijk_args)

        if coeff == 1:
            coeff = np.ones(len(theta0_rad))

        temp_f_coord_idxs = []
        for i in range(self.n_atoms - 2):
            # assign coordinates to interaction/derivative 
            xi_idxs = np.arange(n_args) + i*3
            temp_f_coord_idxs.append(xi_idxs)

        # Add a couple gaussians along each bond
        for n in range(len(theta0_rad)):
            # gaussian well at position theta0_rad[m]
            f_sym = coeff[n]*sympy.exp(-kappa[n]*sympy.cos(self.theta_ijk_sym - theta0_rad[n]))
            f_lamb = sympy.lambdify(self.theta_ijk_args, f_sym, modules="numpy")

            temp_df_funcs = []
            temp_d2f_funcs = []
            for i in range(len(self.theta_ijk_args)):
                # take derivative wrt argument n
                df_sym = f_sym.diff(self.theta_ijk_args[i])
                d2f_sym = df_sym.diff(self.theta_ijk_args[i])
                temp_df_funcs.append(sympy.lambdify(self.theta_ijk_args, df_sym, modules="numpy"))
                temp_d2f_funcs.append(sympy.lambdify(self.theta_ijk_args, d2f_sym, modules="numpy"))

            self._add_test_functions(f_sym, f_lamb, temp_f_coord_idxs, temp_df_funcs, temp_d2f_funcs)

    def gaussian_pair_test_funcs(self, r0_nm, w_nm, coeff=1, bond_cutoff=3):
        """Assign a Gaussian well a each position"""

        assert self.n_atoms > bond_cutoff, "Not enough number of atoms to have angle interactions"
        n_args = len(self.rij_args)

        if coeff == 1:
            coeff = np.ones(len(r0_nm))

        # Add a couple gaussians along each bond
        for n in range(len(r0_nm)):
            # gaussian well at position r0_nm[m]
            f_sym = coeff[n]*sympy.exp(-self.one_half*((self.r12_sym - r0_nm[n])/w_nm[n])**2)
            f_lamb = sympy.lambdify(self.rij_args, f_sym, modules="numpy")

            temp_f_coord_idxs = self._generate_pairwise_idxs(bond_cutoff=bond_cutoff)

            temp_df_funcs = []
            temp_d2f_funcs = []
            for i in range(len(self.rij_args)):
                # take derivative wrt argument n
                df_sym = f_sym.diff(self.rij_args[i])
                d2f_sym = df_sym.diff(self.rij_args[i])
                temp_df_funcs.append(sympy.lambdify(self.rij_args, df_sym, modules="numpy"))
                temp_d2f_funcs.append(sympy.lambdify(self.rij_args, d2f_sym, modules="numpy"))

            self._add_test_functions(f_sym, f_lamb, temp_f_coord_idxs, temp_df_funcs, temp_d2f_funcs)
            
    def gaussian_cv_test_funcs(self, cv_r0, cv_w):
        """Add Gaussian test functions in collective variable space
        
        Parameters
        ----------
        cv_r0 : np.ndarray(P_cv, M) 
            Centers of test functions in collective variable space

        cv_w : np.ndarray(P_cv)
            Widths of test functions in collective variable space
        """

        # 1. Test functions as function of TICs
        #   a. Gaussian centers and widths


        for n in range(len(cv_r0)):
            # test function is Gaussian with center r0 and width w
            f_sym = (self.cv_sym[0] - cv_r0[n][0])**2
            for i in range(1, self.n_cv_dim):
                f_sym += (self.cv_sym[i] - cv_r0[n][i])**2
            f_sym = sympy.exp(-self.one_half*f_sym/(cv_w[n]**2))
            f_lamb = sympy.lambdify(self.cv_args, f_sym, modules="numpy")

            self.cv_f_sym.append(f_sym)
            self.cv_f_funcs.append(f_lamb)

            # first and second derivative wrt each arg
            temp_cv_df_funcs = []
            temp_cv_d2f_funcs = []
            for i in range(len(self.cv_args)):
                df_sym = f_sym.diff(self.cv_args[i])
                temp_cv_df_funcs.append(sympy.lambdify(self.cv_args, df_sym, modules="numpy"))

                # need all mixed second derivatives of test functions
                temp_d2f_funcs = []
                for j in range(len(self.cv_args)):
                    d2f_sym = df_sym.diff(self.cv_args[j])
                    temp_d2f_funcs.append(sympy.lambdify(self.cv_args, d2f_sym, modules="numpy"))
                temp_cv_d2f_funcs.append(temp_d2f_funcs)

            self.cv_df_funcs.append(temp_cv_df_funcs)
            self.cv_d2f_funcs.append(temp_cv_d2f_funcs)

    ##################################################
    # EVALUATE GRADIENT OF POTENTIAL
    ##################################################
    def U_cv_potential(self, coeff, cv_vals):

        U = np.zeros(len(cv_vals))
        for i in range(len(coeff)):
            U += coeff[i]*Ucg.cv_U_funcs[i](cv_vals)
        U -= U.min()
        return U


    def potential_U0(self, xyz_traj, cv_traj, sumterms=True):
        """Fixed potential energy term
        
        Parameters
        ----------
        xyz_traj : 

        cv_traj : np.ndarray
            Collective variable trajectory.
        
        Returns
        -------
        U0 : np.ndarray
            Potential 
        """

        if sumterms:
            U0 = np.zeros(xyz_traj.shape[0], float)
            for i in range(len(self.U_funcs[0])): 
                coord_idxs = self.U_coord_idxs[0][i]
                U_func = self.U_funcs[0][i]
                
                for n in range(len(coord_idxs)):
                    # coordinates assigned this potential
                    U0 += U_func(*xyz_traj[:,coord_idxs[n]].T)
        else:
            U0 = []
            for i in range(len(self.U_funcs[0])): 
                coord_idxs = self.U_coord_idxs[0][i]
                U_func = self.U_funcs[0][i]

                Uterm = np.zeros(xyz_traj.shape[0], float)
                for n in range(len(coord_idxs)):
                    # coordinates assigned this potential
                    Uterm += U_func(*xyz_traj[:,coord_idxs[n]].T)
                U0.append(Uterm)
        return U0

    def gradient_U0(self, xyz_traj, cv_traj):
        """Gradient of fixed potential terms
        
        Parameters
        ----------
        traj : mdtraj.Trajectory
            
        Returns
        -------
        grad_U0 : np.ndarray
            Matrix of gradients
        """

        grad_U0 = np.zeros((xyz_traj.shape[0], self.n_dof), float)

        for i in range(len(self.U_funcs[0])): 
            # parameter i corresponds to functional form i
            # coords assigned to functional form i
            coord_idxs = self.U_coord_idxs[0][i]    # I_r

            for j in range(len(self.dU_funcs[0][i])):
                # derivative wrt argument j
                d_func = self.dU_funcs[0][i][j]
                
                for n in range(len(coord_idxs)):
                    # coordinates assigned to this derivative
                    deriv = d_func(*xyz_traj[:,coord_idxs[n]].T)

                    # derivative is wrt to coordinate index dxi
                    dxi = coord_idxs[n][j]

                    # force on each coordinate is separated by associated
                    # parameter
                    grad_U0[:, dxi] += deriv
        return grad_U0

    def potential_U1(self, xyz_traj, cv_traj):
        """Potential energy associated with each parameter
        
        Parameters
        ----------
        xyz_traj : 
            Cartesian coordinate trajectory.

        cv_traj : np.ndarray
            Collective variable trajectory.
        
        Returns
        -------
        U1 : np.ndarray
            Potential energy for each parameter
        """

        if self.using_cv:
            U1 = np.zeros((xyz_traj.shape[0], self.n_params), float)
            for i in range(self.n_params): 
                # derivative wrt argument j
                U_func = self.cv_U_funcs[i]
                U1[:,i] = U_func(*cv_traj.T)
        else:
            raise NotImplementedError
        return U1

    def gradient_U1(self, xyz_traj, cv_traj):
        """Gradient of potential form associated with each parameter
        
        Parameters
        ----------
        xyz_traj : 

        cv_traj : np.ndarray
            Collective variable trajectory.
        
        Returns
        -------
        grad_x_U1 : np.ndarray
            Gradient with respect to Cartesian coordinates
        """

        if self.using_cv:
            # if potentials depend on collective variables
            # the gradient requires chain rule 
            Jac = self._cv_cartesian_Jacobian(xyz_traj)

            grad_cv_U1 = np.zeros((xyz_traj.shape[0], self.n_cv_dim, self.n_params), float)
            for i in range(self.n_params): 
                for j in range(len(self.cv_dU_funcs[i])):
                    # derivative wrt argument j
                    d_func = self.cv_dU_funcs[i][j]
                    grad_cv_U1[:,j,i] = d_func(*cv_traj.T)

            grad_x_U1 = np.einsum("tnd,tnr->tdr", Jac, grad_cv_U1)
        else:
            # gradient with respect to Cartesian coordinates, directly. 
            grad_x_U1 = np.zeros((xyz_traj.shape[0], self.n_dof, self.n_params), float)

            for i in range(self.n_params): 
                # parameter i corresponds to functional form i
                # coords assigned to functional form i
                coord_idxs = self.U_coord_idxs[1][i]    # I_r

                for j in range(len(self.dU_funcs[1][i])):
                    # derivative wrt argument j
                    d_func = self.dU_funcs[1][i][j]
                    
                    for n in range(len(coord_idxs)):
                        # coordinates assigned to this derivative
                        deriv = d_func(*xyz_traj[:,coord_idxs[n]].T)

                        # derivative is wrt to coordinate index dxi
                        dxi = coord_idxs[n][j]

                        # force on each coordinate is separated by associated
                        # parameter
                        grad_x_U1[:, dxi, i] += deriv
        return grad_x_U1

    #########################################################
    # EVALUATE TEST FUNCTIONS
    #########################################################
    def test_functions(self, xyz_traj, cv_traj):
        """Test functions"""

        test_f = np.zeros((xyz_traj.shape[0], self.n_test_funcs), float)

        if self.using_cv:
            # collective variable test functions
            for i in range(len(self.cv_f_funcs)):
                test_f[:,i] = self.cv_f_funcs[i](*cv_traj.T)
        else:
            # Cartesian coordinate test functions 
            start_idx = 0
            for j in range(len(self.f_funcs)):
                # test function form j
                f_j_func = self.f_funcs[j]

                for n in range(len(self.f_coord_idxs[j])):
                    # each coordinate assignment is a different test function
                    xi_idxs = self.f_coord_idxs[j][n]
                    test_f[:, start_idx + n] = f_j_func(*xyz_traj[:,xi_idxs].T)
                start_idx += len(self.f_coord_idxs[j])

        return test_f
            
    def test_funcs_gradient_and_laplacian(self, xyz_traj, cv_traj):
        """Gradient of test functions"""

        if self.using_cv:
            # collective variable test functions
            Jac = self._cv_cartesian_Jacobian(xyz_traj)
            Hess_cv = self._cv_cartesian_Hessian(xyz_traj)
            Hess_f = self._Hessian_test_func_cv(cv_traj)
            grad_cv_f = self._gradient_test_functions_cv_wrt_cv(cv_traj)
            One = np.ones(self.n_dof)

            grad_x_f = np.einsum("tmd,tmp->tdp", Jac, grad_cv_f)

            Lap_term1 = np.einsum("d,tkd,tkp->tp", One, Hess_cv, grad_cv_f)
            Lap_term2 = np.einsum("tnd,tmd,tmnp->tp", Jac, Jac, Hess_f)   
            Lap_x_f = Lap_term1 + Lap_term2
        else:
            # Cartesian coordinate test functions 
            grad_x_f = self._cartesian_test_funcs_gradient(xyz_traj)
            Lap_x_f = self._cartesian_test_funcs_laplacian(xyz_traj)

        return grad_x_f, Lap_x_f

    def _cartesian_test_funcs_gradient(self, xyz_traj):
        grad_x_f = np.zeros((xyz_traj.shape[0], self.n_dof, self.n_test_funcs), float)
        
        start_idx = 0
        for j in range(len(self.f_funcs)):
            for n in range(len(self.f_coord_idxs[j])):
                # coordinates assigned to function j
                xi_idxs = self.f_coord_idxs[j][n]

                # innermost loop calculates components of grad f_j({x}_n)
                for i in range(len(self.df_funcs[j])):
                    # first derivative wrt argument i
                    dxi = xi_idxs[i] 

                    df_j_func = self.df_funcs[j][i]
                    grad_x_f[:, dxi, start_idx + n] = df_j_func(*xyz_traj[:,xi_idxs].T)

            start_idx += len(self.f_coord_idxs[j])
        return grad_x_f

    def _cartesian_test_funcs_laplacian(self, xyz_traj):
        """Laplacian of test functions"""

        #n_test_funcs = np.sum([ len(self.f_coord_idxs[i]) for i in range(len(self.f_funcs)) ])

        Lap_x_f = np.zeros((xyz_traj.shape[0], self.n_test_funcs), float)

        start_idx = 0
        for j in range(len(self.f_funcs)):
            for n in range(len(self.f_coord_idxs[j])):
                # coordinates assigned to function j
                xi_idxs = self.f_coord_idxs[j][n]

                for i in range(len(self.d2f_funcs[j])):
                    # add the double derivative wrt argument i
                    d2f_j_func = self.d2f_funcs[j][i]
                    Lap_x_f[:, start_idx + n] += d2f_j_func(*xyz_traj[:,xi_idxs].T)

            start_idx += len(self.f_coord_idxs[j])

        return Lap_x_f

    #########################################################################
    # Collective variable test functions. Gradient and Laplacian
    #########################################################################
    def calculate_cv(self, xyz_traj):
        """Value of collective variables"""

        # collective variables are sum of features
        Vals_cv = np.zeros((xyz_traj.shape[0], self.n_cv_dim), float)
        for m in range(self.n_feature_types):
            # loop over feature types
            chi_m = self.chi_funcs[m]

            for i in range(len(self.chi_coord_idxs[m])):
                # coefficients of this feature to each cv
                b_coeff = self.chi_coeff[m][i,:]
                chi_avg = self.chi_mean[m][i]
                xi_idxs = self.chi_coord_idxs[m][i]

                mean_free_chi = chi_m(*xyz_traj[:,xi_idxs].T) - chi_avg
                Vals_cv[:,:] += np.einsum("m,t->tm", b_coeff, mean_free_chi)

        return Vals_cv

    def _cv_cartesian_Jacobian(self, xyz_traj):
        """Gradient of features wrt Cartesian coordinates"""

        # partial derivative of CV wrt to cartesian coordinates
        Jacobian = np.zeros((xyz_traj.shape[0], self.n_cv_dim, self.n_dof), float)

        for m in range(self.n_feature_types):
            # loop over feature types

            for i in range(len(self.chi_coord_idxs[m])):
                # coefficients of this feature to each cv
                b_coeff = self.chi_coeff[m][i,:]
                xi_idxs = self.chi_coord_idxs[m][i]

                # add together gradients of each feature 
                for n in range(len(xi_idxs)):
                    dxi = xi_idxs[n]
                    d_chi = self.dchi_funcs[m][n](*xyz_traj[:,xi_idxs].T)
                    Jacobian[:,:,dxi] += np.einsum("m,t -> tm", b_coeff, d_chi)
        return Jacobian

    def _cv_cartesian_Hessian(self, xyz_traj):
        """Second derivative of features wrt Cartesian coords"""

        # partial derivative of CV wrt to cartesian coordinates
        Hess_cv = np.zeros((xyz_traj.shape[0], self.n_cv_dim, self.n_dof), float)

        for m in range(self.n_feature_types):
            # loop over feature types

            for i in range(len(self.chi_coord_idxs[m])):
                # coefficients of this feature to each cv
                b_coeff = self.chi_coeff[m][i,:]
                xi_idxs = self.chi_coord_idxs[m][i]

                # add together gradients of each feature 
                for n in range(len(xi_idxs)):
                    dxi = xi_idxs[n]
                    d2_chi = self.d2chi_funcs[m][n](*xyz_traj[:,xi_idxs].T)
                    Hess_cv[:,:,dxi] += np.einsum("m,t -> tm", b_coeff, d2_chi)
        return Hess_cv

    def _Hessian_test_func_cv(self, cv_traj):

        Hess_f = np.zeros((cv_traj.shape[0], self.n_cv_dim, self.n_cv_dim, self.n_test_funcs), float)

        for m in range(self.n_test_funcs):
            for i in range(self.n_cv_dim):
                for j in range(self.n_cv_dim):
                    d2_test_func = self.cv_d2f_funcs[m][i][j]
                    Hess_f[:,i,j,m] = d2_test_func(*cv_traj.T)
        return Hess_f

    def _gradient_test_functions_cv_wrt_cv(self, cv_traj):

        grad_cv_f = np.zeros((cv_traj.shape[0], self.n_cv_dim, self.n_test_funcs), float)
        for i in range(len(self.cv_f_funcs)):
            # for each test functions we have the derivative wrt each argument
            for j in range(len(self.cv_df_funcs[i])):
                d_func = self.cv_df_funcs[i][j]
                grad_cv_f[:,j,i] = d_func(*cv_traj.T)
        return grad_cv_f

    ###############################################################
    # CALCULATE MATRICES FOR EIGENPAIR METHOD
    ###############################################################

    def _generator_scalar_products(self, coeff, trajnames, topfile, psinames, M=1, cv_names=[], verbose=False, a_coeff=None):

        # DEPRECATED
        P = self.n_test_funcs

        if a_coeff is None:
            fixed_a = False
            c_r = coeff[:-1] 
            a_coeff = 1./coeff[-1]  #?
        else:
            fixed_a = True
            c_r = coeff

        # if constant diff coeff
        if self.constant_a_coeff:
            psi_fj = np.zeros((M, P), float)
            psi_gU0_fj = np.zeros((M, P), float)
            psi_gU1_fj = np.zeros((M, P), float)
            psi_Lap_fj = np.zeros((M, P), float)
            psi_Gen_fj = np.zeros((M, P), float)
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if self.using_cv and not self.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        N_prev = 0
        for n in range(len(trajnames)):
            if verbose:
                if n == len(trajnames) - 1:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d} DONE".format(n + 1, len(trajnames)))
                else:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d}".format(n + 1, len(trajnames)), end="\r")
                sys.stdout.flush()
            # load eigenvectors
            psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

            if len(cv_names) > 0:
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in cv_names[n] ]).T
            else:
                cv_traj = None

            start_idx = 0
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                N_chunk = chunk.n_frames

                psi_chunk = psi_traj[start_idx:start_idx + N_chunk,:]

                if cv_traj is None:
                    cv_chunk = self.calculate_cv(chunk)
                else:
                    cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_chunk, self.n_dof))

                # calculate gradient of fixed and parametric potential terms
                grad_U1 = self.gradient_U1(xyz_traj, cv_chunk)
                grad_U1 = np.einsum("r,tdr->td", c_r, grad_U1)

                # calculate test function values, gradient, and Laplacian
                test_f = self.test_functions(xyz_traj, cv_chunk)
                grad_f, Lap_f = self.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

                if self.using_U0:
                    grad_U0 = self.gradient_U0(xyz_traj, cv_chunk)
                    curr_psi_gradU0_fj = np.einsum("tm, td,tdp->mp", psi_chunk, -a_coeff*grad_U0, grad_f)
                    psi_gU0_fj = (curr_psi_gradU0_fj + float(N_prev)*psi_gU0_fj)/(float(N_prev + N_chunk))

                # calculate generator terms
                curr_psi_fj = np.einsum("tm,tp->mp", psi_chunk, test_f)
                curr_psi_gradU1_fj = np.einsum("tm,td,tdp->mp", psi_chunk, -a_coeff*grad_U1, grad_f)
                curr_psi_Lap_fj = np.einsum("tm,tp->mp", psi_chunk, (a_coeff/self.beta)*Lap_f)

                psi_fj = (curr_psi_fj + float(N_prev)*psi_fj)/(float(N_prev + N_chunk))
                psi_gU1_fj = (curr_psi_gradU1_fj + float(N_prev)*psi_gU1_fj)/(float(N_prev + N_chunk))
                psi_Lap_fj = (curr_psi_Lap_fj + float(N_prev)*psi_Lap_fj)/(float(N_prev + N_chunk))

                start_idx += N_chunk
                N_prev += N_chunk

        self._psi_fj = psi_fj
        self._psi_gU0_fj = psi_gU0_fj
        self._psi_gU1_fj = psi_gU1_fj
        self._psi_Lap_fj = psi_Lap_fj
        self._psi_Gen_fj = psi_gU0_fj + psi_gU1_fj + psi_Lap_fj 

    def _eigenpair_Lap_f(self, coeff, trajnames, topfile, psinames, M=1, cv_names=[], verbose=False):

        P = self.n_test_funcs

        # if constant diff coeff
        if self.constant_a_coeff:
            c_r = coeff[:-1] 
            D_coeff = 1./coeff[-1]
            Lap_fj = np.zeros(P, float)
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if self.using_cv and not self.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(trajnames):
            raise ValueError("Need eigenvector for every trajectory!")


        N_prev = 0
        for n in range(len(trajnames)):
            if verbose:
                if n == len(trajnames) - 1:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d} DONE".format(n + 1, len(trajnames)))
                else:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d}".format(n + 1, len(trajnames)), end="\r")
                sys.stdout.flush()
            # load eigenvectors
            psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

            if len(cv_names) > 0:
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in cv_names[n] ]).T
            else:
                cv_traj = None

            start_idx = 0
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                N_chunk = chunk.n_frames

                psi_chunk = psi_traj[start_idx:start_idx + N_chunk,:]

                if cv_traj is None:
                    cv_chunk = self.calculate_cv(chunk)
                else:
                    cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_chunk, self.n_dof))

                # calculate test function Laplacian
                grad_f, Lap_f = self.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)
                curr_Lap_fj = np.sum((D_coeff/self.beta)*Lap_f)
                Lap_fj = (curr_Lap_fj + float(N_prev)*Lap_fj)/(float(N_prev + N_chunk))

                start_idx += N_chunk
                N_prev += N_chunk

        self.eigenpair_Lap_fj = Lap_fj

    #def _eigenpair_Lap_f(self, trajnames, topfile, psinames, psi_bin_edges, ti_file, M=1, cv_names=[], verbose=False):
    #    """Calculate Laplacian of test functions 
    #   
    #    Parameters
    #    ----------
    #    trajnames : list, str
    #        Trajectory filenames

    #    topfile : str
    #        Filename for topology (pdb)

    #    psinames : list, str
    #        Filenames for 

    #    psi_bin_edges : list, np.ndarray
    #        Bin edges for histograming along eigenvector
    #        
    #    ti_file : str
    #        Filename for timescales

    #    M : int (default=1)
    #        Number of timescales to keep in eigenfunction expansion.
    #    
    #    cv_names : list, str (opt)
    #        Collective variable rilenames if pre-calculated. Will calculate
    #        collective variables on the fly if not given. 

    #    """
    #    # TODO: allow collective variable to be different than eigenvector

    #    P = self.n_test_funcs
    #    n_psi_bins = len(psi_bin_edges) - 1

    #    # if constant diff coeff
    #    if self.constant_a_coeff:
    #        avg_Lapf = np.zeros((M, P, n_psi_bins), float)
    #    else:
    #        raise NotImplementedError("Only constant diffusion coefficient is supported.")

    #    if self.using_cv and not self.cv_defined:
    #        raise ValueError("Collective variables are not defined!")

    #    if len(psinames) != len(trajnames):
    #        raise ValueError("Need eigenvector for every trajectory!")

    #    kappa = 1./np.load(ti_file)[:M]

    #    N_prev = np.zeros((P, n_psi_bins))
    #    for n in range(len(trajnames)):
    #        if verbose:
    #            if n == len(trajnames) - 1:
    #                print("Lap f_j: {:>5d}/{:<5d} DONE".format(n + 1, len(trajnames)))
    #            else:
    #                print("Lap f_j: {:>5d}/{:<5d}".format(n + 1, len(trajnames)), end="\r")
    #            sys.stdout.flush()
    #        # load eigenvectors
    #        psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

    #        if len(cv_names) > 0:
    #            cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in cv_names[n] ]).T
    #        else:
    #            cv_traj = None

    #        start_idx = 0
    #        for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
    #            N_chunk = chunk.n_frames

    #            psi_chunk = psi_traj[start_idx:start_idx + N_chunk,:]

    #            if cv_traj is None:
    #                cv_chunk = self.calculate_cv(chunk)
    #            else:
    #                cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

    #            # cartesian coordinates unraveled
    #            xyz_traj = np.reshape(chunk.xyz, (N_chunk, self.n_dof))

    #            # calculate test Laplacian
    #            grad_f, Lap_f = self.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

    #            # bin along eigenvectors
    #            for j in range(P):
    #                curr_Lap_fj = bin1d(psi_chunk[:,0], Lap_f[:,j], bins=psi_bin_edges, statistic="sum")[0]
    #                N_chunk_j = bin1d(psi_chunk[:,0], Lap_f[:,j], bins=psi_bin_edges, statistic="count")[0]

    #                N_tot_j = N_prev[j] + N_chunk_j
    #                nonzero_bins = N_tot_j > 0
    #                if np.any(nonzero_bins): 
    #                    #import pdb
    #                    #pdb.set_trace()
    #                    avg_Lapf[0, j, nonzero_bins] = (curr_Lap_fj[nonzero_bins] + float(N_prev[j, nonzero_bins])*avg_Lapf[0, j, nonzero_bins])/float(N_tot_j[nonzero_bins])

    #                N_prev[j] += N_chunk_j
    #            start_idx += N_chunk

    #    self.eigenpair_Lapf_vs_psi = avg_Lapf

    def _eigenpair_Jacobian(self, trajnames, topfile, psinames, ti_file, M=1, cv_names=[]):
        """Calculate eigenpair matrices
       
        Parameters
        ----------
        trajnames : list, str
            Trajectory filenames

        topfile : str
            Filename for topology (pdb)

        psinames : list, str
            Filenames for 
            
        ti_file : str
            Filename for timescales

        M : int (default=1)
            Number of timescales to keep in eigenfunction expansion.
        
        cv_names : list, str (opt)
            Collective variable rilenames if pre-calculated. Will calculate
            collective variables on the fly if not given. 

        """
        # TODO: allow collective variable to be different than eigenvector

        JTJ = np.zeros((self.n_cv_dim, self.n_cv_dim), float)

        if self.using_cv and not self.cv_defined:
            raise ValueError("Collective variables are not defined!")

        print("calculating eigenpair Jacobian...")
        N_prev = 0
        for n in range(len(trajnames)):
            print("  traj: " + str(n+1))
            sys.stdout.flush()

            start_idx = 0
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                N_curr = chunk.n_frames

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_curr, self.n_dof))

                Jac = self._cv_cartesian_Jacobian(xyz_traj)
                curr_JTJ = np.einsum("tmd,tnd->tmn", Jac, Jac)
                
                jmin, jmax = curr_JTJ.max()

                # running average to reduce numerical error
                JTJ = (curr_JTJ + float(N_prev)*JTJ)/float(N_prev + N_curr)

                start_idx += N_curr
                N_prev += N_curr

        self.eigenpair_JTJ = JTJ

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

