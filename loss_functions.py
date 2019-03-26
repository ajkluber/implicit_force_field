from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
import scipy.linalg as scl

from scipy.optimize import least_squares
from scipy.optimize import minimize
import scipy.linalg

import mdtraj as md


from memory_profiler import profile
# TODO: nonlinear loss function

class CrossValidatedLoss(object):
    def __init__(self, topfile, trajnames, savedir, n_cv_sets=5):
        """Creates matrices for minimizing the linear spectral loss equations
        
        Parameters
        ----------
        savedir : str
            Where matrices should be saved.
            
        n_cv_sets : opt.
            Number of k-fold cross validation sets to use.
            
        """

        self.topfile = topfile
        self.trajnames = trajnames

        self.savedir = savedir
        self.n_cv_sets = n_cv_sets
        self.cv_sets_are_assigned = False

    def assign_crossval_sets(self):
        """Randomly assign frames to training and validation sets

        Parameters
        ----------
        topfile : str
            Topology filename.

        trajnames : list, str
            List of trajectory filenames.

        """

        set_assignment = []
        traj_n_frames = []
        for n in range(len(self.trajnames)):
            length = 0
            for chunk in md.iterload(self.trajnames[n], top=self.topfile, chunk=1000):
                length += chunk.n_frames
            traj_n_frames.append(length)

            set_assignment.append(np.random.randint(low=0, high=self.n_cv_sets, size=length))
        self.total_n_frames = sum(traj_n_frames)

        self.n_frames_in_set = []
        for k in range(self.n_cv_sets):
            self.n_frames_in_set.append(np.sum([ np.sum(set_assignment[i] == k) for i in range(len(set_assignment)) ]))
        self.set_weights = [ (self.n_frames_in_set[j]/float(self.total_n_frames)) for j in range(self.n_cv_sets) ]

        self.cv_set_assignment = set_assignment
        self.cv_sets_are_assigned = True

    def matrix_files_exist(self):
        X_files_exist = np.all([ os.path.exists("{}/X_{}_{}.npy".format(self.savedir, self.suffix, i + 1)) for i in range(self.n_cv_sets) ])
        d_files_exist = np.all([ os.path.exists("{}/d_{}_{}.npy".format(self.savedir, self.suffix, i + 1)) for i in range(self.n_cv_sets) ])
        set_files_exist = np.all([ os.path.exists("{}/frame_set_{}_{}.npy".format(self.savedir, self.suffix, i + 1)) for i in range(self.n_cv_sets) ])
        files_exist = X_files_exist and d_files_exist and set_files_exist

        return files_exist

    def _save_matrices(self): 
        for k in range(self.n_cv_sets):
            np.save("{}/X_{}_{}.npy".format(self.savedir, self.suffix, k + 1), self.X_sets[k])
            np.save("{}/d_{}_{}.npy".format(self.savedir, self.suffix, k + 1), self.d_sets[k])
            np.save("{}/frame_set_{}_{}.npy".format(self.savedir, self.suffix, k + 1), self.cv_set_assignment[k])    

        np.save("{}/X_{}.npy".format(self.savedir, self.suffix), self.X)
        np.save("{}/d_{}.npy".format(self.savedir, self.suffix), self.d)

    def _load_matrices(self):
        
        if self.n_cv_sets is None:
            raise ValueErro("Need to define number of cross val sets in order to load them")

        print("Loaded saved matrices...")
        self.X_sets = [ np.load("{}/X_{}_{}.npy".format(self.savedir, self.suffix, i + 1)) for i in range(self.n_cv_sets) ]
        self.d_sets = [ np.load("{}/d_{}_{}.npy".format(self.savedir, self.suffix, i + 1)) for i in range(self.n_cv_sets) ]
        set_assignment = [ np.load("{}/frame_set_{}_{}.npy".format(self.savedir, self.suffix, i + 1)) for i in range(self.n_cv_sets) ]

        self.n_frames_in_set = []
        for k in range(self.n_cv_sets):
            self.n_frames_in_set.append(np.sum([ np.sum(set_assignment[i] == k) for i in range(self.n_cv_sets) ]))
        self.total_n_frames = np.sum([ set_assignment[i].shape[0] for i in range(self.n_cv_sets) ])
        self.set_weights = [ (self.n_frames_in_set[j]/float(self.total_n_frames)) for j in range(self.n_cv_sets) ]

        if self.n_cv_sets > 1:
            self.X = np.sum([ self.set_weights[j]*self.X_sets[j] for j in range(self.n_cv_sets) ], axis=0)
            self.d = np.sum([ self.set_weights[j]*self.d_sets[j] for j in range(self.n_cv_sets) ], axis=0)
        else:
            self.X = self.X_sets[0]
            self.d = self.d_sets[0]

        self.matrices_estimated = True
        self.cv_sets_are_assigned = True
        self._training_and_validation_matrices()

    def _training_and_validation_matrices(self):
        """Create training and validation matrices"""

        self.X_train_val = []
        self.d_train_val = []
        for i in range(self.n_cv_sets):
            frame_subtotal = self.total_n_frames - self.n_frames_in_set[i]
            #frame_subtotal = np.sum([ n_frames_in_set[j] for j in range(self.n_cv_sets) if j != i ])

            train_X = []
            train_d = []
            for j in range(self.n_cv_sets):
                w_j = self.n_frames_in_set[j]/float(frame_subtotal)
                if j != i:
                    train_X.append(w_j*self.X_sets[j])
                    train_d.append(w_j*self.d_sets[j])

            train_X = np.sum(np.array(train_X), axis=0)
            train_d = np.sum(np.array(train_d), axis=0)

            self.X_train_val.append([ train_X, self.X_sets[i]])
            self.d_train_val.append([ train_d, self.d_sets[i]])

    def _solve_regularized(self, alphas, X_train_val, d_train_val, X, d, D):

        coeffs = [] 
        train_mse = []
        valid_mse = []
        for i in range(len(alphas)):
            if i == len(alphas) - 1: 
                print("Solving: {:>5d}/{:<5d} DONE".format(i+1, len(alphas)))
            else:
                print("Solving: {:>5d}/{:<5d}".format(i+1, len(alphas)), end="\r")
            sys.stdout.flush()
            
            # folds are precalculated matrices on trajectory chunks
            train_mse_folds = [] 
            valid_mse_folds = [] 
            for k in range(self.n_cv_sets):
                X_train, X_val = X_train_val[k]
                d_train, d_val = d_train_val[k]

                X_reg = np.dot(X_train.T, X_train) + alphas[i]*D
                d_reg = np.dot(X_train.T, d_train)
                coeff_fold = scl.lstsq(X_reg, d_reg, cond=1e-10)[0]
        
                train_mse_folds.append(np.mean((np.dot(X_train, coeff_fold) - d_train)**2))
                valid_mse_folds.append(np.mean((np.dot(X_val, coeff_fold) - d_val)**2))
        
            train_mse.append([np.mean(train_mse_folds), np.std(train_mse_folds)/np.sqrt(float(self.n_cv_sets))])
            valid_mse.append([np.mean(valid_mse_folds), np.std(valid_mse_folds)/np.sqrt(float(self.n_cv_sets))])

            X_reg = np.dot(X.T, X) + alphas[i]*D
            d_reg = np.dot(X.T, d)
            coeffs.append(scl.lstsq(X_reg, d_reg, cond=1e-10)[0])

        return np.array(coeffs), np.array(train_mse), np.array(valid_mse)

    def solve(self, alphas, method="ridge"):
        """Ridge regression with cross-validation
        
        Parameters
        ----------
        alphas : np.array
            Regularization parameter values.

        method : str
            Regularization method to use.
            
        """

        if not self.matrices_estimated:
            raise ValueError("Need to calculate or eigenpair matrices")

        if not hasattr(self, "X_train_val"):
            self._training_and_validation_matrices()

        if method == "ridge":
            D = np.identity(self.X.shape[1])
        else:
            raise ValueError("Only method=ridge supported")

        coeffs, train_mse, valid_mse = self._solve_regularized(alphas, self.X_train_val, self.d_train_val, self.X, self.d, D)

        self.alphas = alphas
        self.coeffs = coeffs
        self.train_mse = train_mse
        self.valid_mse = valid_mse

        self.alpha_star_idx = np.argmin(self.valid_mse[:,0])
        self.coeff_star = self.coeffs[self.alpha_star_idx]

    def solve_with_fixed_params(self, alphas, fix_ck_idxs, fix_ck_vals, method="ridge"):
        """Ridge regression with cross-validation
        
        Parameters
        ----------
        alphas : np.array
            Regularization parameter values.

        fix_ck_idxs : np.array
            Indices of parameters to fix

        fix_ck_vals : np.array
            Values of fixed parameters

        method : str
            Regularization method to use.
            
        """

        if not self.matrices_estimated:
            raise ValueError("Need to calculate or eigenpair matrices")

        if not hasattr(self, "X_train_val"):
            self._training_and_validation_matrices()

        keep_idxs = np.array([ x for x in range(self.X.shape[1]) if x not in fix_ck_idxs ])

        X_train_val = []
        d_train_val = []
        for k in range(self.n_cv_sets):
            X_train, X_val = np.copy(self.X_train_val[k])
            d_train, d_val = np.copy(self.d_train_val[k])

            d_train -= np.einsum("ij,j->i", X_train[:,fix_ck_idxs], fix_ck_vals)
            d_val -= np.einsum("ij,j->i", X_val[:,fix_ck_idxs], fix_ck_vals)

            X_train_val.append([X_train[:,keep_idxs], X_val[:,keep_idxs]])
            d_train_val.append([d_train, d_val])

        X = np.copy(self.X[:,keep_idxs])
        d = np.copy(self.d - np.einsum("ij,j->i", self.X[:,fix_ck_idxs], fix_ck_vals))

        if method == "ridge":
            D = np.identity(X.shape[1])
        else:
            raise ValueError("Only method=ridge supported")

        coeffs, train_mse, valid_mse = self._solve_regularized(alphas, X_train_val, d_train_val, X, d, D)
        return coeffs, train_mse, valid_mse

class LinearSpectralLoss(CrossValidatedLoss):

    def __init__(self, topfile, trajnames, savedir, n_cv_sets=5, recalc=False):
        """Creates matrices for minimizing the linear spectral loss equations
        
        Parameters
        ----------
        savedir : str
            Where matrices should be saved.
            
        n_cv_sets : opt.
            Number of k-fold cross validation sets to use.
            
        recalc : bool
            Recalculated 

        """
        CrossValidatedLoss.__init__(self, topfile, trajnames, savedir, n_cv_sets=n_cv_sets)

        self.suffix = "EG"
        self.matrices_estimated = False
        self.recalc = recalc

        if self.matrix_files_exist() and not recalc:
            self._load_matrices()

    def calc_matrices(self, Ucg, psinames, ti_file, M=1, coll_var_names=None, verbose=True, include_trajs=[]):
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
        
        coll_var_names : list, str (opt)
            Collective variable rilenames if pre-calculated. Will calculate
            collective variables on the fly if not given. 

        """

        self.Ucg = Ucg

        if self.n_cv_sets is None:
            self.n_cv_sets = 1
        else:
            if self.n_cv_sets > 1 and not self.cv_sets_are_assigned:
                self.assign_crossval_sets()

        R = Ucg.n_tot_params
        P = Ucg.n_test_funcs

        # if constant diff coeff
        if Ucg.constant_a_coeff:
            d = np.zeros((self.n_cv_sets, P), float)
            if Ucg.fixed_a_coeff:
                X = np.zeros((self.n_cv_sets, P, R), float)
                D2 = np.zeros((self.n_cv_sets, R, R), float)    # TODO: high-dimensional smoothness 
            else:
                X = np.zeros((self.n_cv_sets, P, R + 1), float)
                D2 = np.zeros((self.n_cv_sets, R + 1, R + 1), float)    # TODO: high-dimensional smoothness 
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if Ucg.using_cv and not Ucg.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(self.trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        kappa = 1./np.load(ti_file)[:M]

        #if Ucg.fixed_a_coeff:
        #    beta_kappa_const = Ucg.beta*kappa/Ucg.a_coeff
        #else:
        #    beta_kappa_const = Ucg.beta*kappa

        if len(include_trajs) > 0:
            n_trajs = len(include_trajs)
        else:
            n_trajs = len(self.trajnames)

        count = 0
        N_prev = np.zeros(self.n_cv_sets, float)
        for n in range(len(self.trajnames)):
            if n in include_trajs:
                count += 1
            else:
                continue

            if verbose:
                if count == n_trajs:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d} DONE".format(count, n_trajs))
                else:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d}".format(count, n_trajs), end="\r")
                sys.stdout.flush()

            # load eigenvectors
            psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

            if len(coll_var_names) > 0:
                # load collective variable if given
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in coll_var_names[n] ]).T
            else:
                cv_traj = None

            # calculate matrix for trajectory
            start_idx = 0
            for chunk in md.iterload(self.trajnames[n], top=self.topfile, chunk=1000):
                N_chunk = chunk.n_frames

                psi_chunk = psi_traj[start_idx:start_idx + N_chunk,:]

                # generator quantities

                if cv_traj is None:
                    cv_chunk = Ucg.calculate_cv(chunk)
                else:
                    cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_chunk, Ucg.n_dof))

                # calculate gradient of fixed and parametric potential terms
                grad_U1 = Ucg.gradient_U1(xyz_traj, cv_chunk)

                # calculate test function values, gradient, and Laplacian
                test_f = Ucg.test_functions(xyz_traj, cv_chunk)
                grad_f, Lap_f = Ucg.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

                if Ucg.fixed_a_coeff:
                    grad_U1 *= Ucg.a_coeff
                    Lap_f *= Ucg.a_coeff
                    #test_f *= beta_kappa_const

                if Ucg.using_U0:
                    grad_U0 = Ucg.gradient_U0(xyz_traj, cv_chunk)
                    if Ucg.fixed_a_coeff:
                        grad_U0 *= Ucg.a_coeff

                if Ucg.using_D2:
                    #TODO
                    pass

                if self.n_cv_sets == 1: 
                    # dot products with eigenvectors
                    curr_X1 = np.einsum("tm,tdr,tdp->mpr", psi_chunk, -grad_U1, grad_f).reshape((M*P, R))
                    curr_X2 = np.einsum("m,tm,tp->mp", kappa, psi_chunk, test_f).reshape(M*P)

                    curr_d = (-1./Ucg.beta)*np.einsum("tm,tp->mp", psi_chunk, Lap_f).reshape(M*P)

                    if Ucg.using_U0:
                        curr_d += np.einsum("tm,td,tdp->mp", psi_chunk, grad_U0, grad_f).reshape(M*P)

                    if Ucg.fixed_a_coeff:
                        # running average to reduce numerical error
                        curr_d -= curr_X2

                        X[0,:] = (curr_X1 + N_prev[0]*X[0,:,:-1])/(N_prev[0] + N_chunk)
                        d[0,:] = (curr_d + N_prev[0]*d[0,:])/(N_prev[0] + N_chunk)
                    else:
                        # running average to reduce numerical error
                        X[0,:,:-1] = (curr_X1 + N_prev[0]*X[0,:,:-1])/(N_prev[0] + N_chunk)
                        X[0,:,-1] = (curr_X2 + N_prev[0]*X[0,:,-1])/(N_prev[0] + N_chunk)
                        d[0,:] = (curr_d + N_prev[0]*d[0,:])/(N_prev[0] + N_chunk)

                    N_prev[0] += float(N_chunk)
                else:
                    for k in range(self.n_cv_sets):   
                        frames_in_this_set = self.cv_set_assignment[n][start_idx:start_idx + N_chunk] == k

                        if np.any(frames_in_this_set):
                            N_curr_set = np.sum(frames_in_this_set)

                            # average subset of frames for set k
                            psi_subset = psi_chunk[frames_in_this_set]
                            gU1_subset = -grad_U1[frames_in_this_set]
                            gradf_subset = grad_f[frames_in_this_set]
                            testf_subset = test_f[frames_in_this_set]
                            Lap_f_subset = Lap_f[frames_in_this_set]

                            # dot products with eigenvectors
                            curr_X1 = np.einsum("tm,tdr,tdp->mpr", psi_subset, gU1_subset, gradf_subset).reshape((M*P, R))
                            curr_X2 = np.einsum("m,tm,tp->mp", kappa, psi_subset, testf_subset).reshape(M*P)

                            curr_d = (-1./Ucg.beta)*np.einsum("tm,tp->mp", psi_subset, Lap_f_subset).reshape(M*P)

                            if Ucg.using_U0:
                                gU0_subset = grad_U0[frames_in_this_set]
                                curr_d += np.einsum("tm,td,tdp->mp", psi_subset, gU0_subset, gradf_subset).reshape(M*P)

                            if Ucg.fixed_a_coeff:
                                # running average to reduce numerical error
                                curr_d -= curr_X2

                                X[k,:] = (curr_X1 + N_prev[k]*X[k,:])/(N_prev[k] + N_curr_set)
                                d[k,:] = (curr_d + N_prev[k]*d[k,:])/(N_prev[k] + N_curr_set)
                            else:
                                # running average to reduce numerical error
                                X[k,:,:-1] = (curr_X1 + N_prev[k]*X[k,:,:-1])/(N_prev[k] + N_curr_set)
                                X[k,:,-1] = (curr_X2 + N_prev[k]*X[k,:,-1])/(N_prev[k] + N_curr_set)
                                d[k,:] = (curr_d + N_prev[k]*d[k,:])/(N_prev[k] + N_curr_set)

                            N_prev[k] += float(N_curr_set)
                start_idx += N_chunk

        self.X_sets = X 
        self.d_sets = d
        if self.n_cv_sets > 1:
            self.X = np.sum([ self.set_weights[j]*self.X_sets[j] for j in range(self.n_cv_sets) ], axis=0)
            self.d = np.sum([ self.set_weights[j]*self.d_sets[j] for j in range(self.n_cv_sets) ], axis=0)
        else:
            self.X = self.X_sets[0]
            self.d = self.d_sets[0]

        self.matrices_estimated = True
        self._save_matrices()
        self._training_and_validation_matrices()

    def scalar_product_Gen_fj(self, Ucg, coeff, psinames, M=1, cv_names=[], include_trajs=[]):

        #TODO

        if Ucg.fixed_a_coeff:
            c_r = coeff
            a_coeff = Ucg.a_coeff
        else:
            c_r = coeff[:-1] 
            a_coeff = 1./coeff[-1]  #?

        P = Ucg.n_test_funcs
        if Ucg.constant_a_coeff:
            psi_fj = np.zeros((M, P), float)
            psi_gU0_fj = np.zeros((M, P), float)
            psi_gU1_fj = np.zeros((M, P), float)
            psi_Lap_fj = np.zeros((M, P), float)
            psi_Gen_fj = np.zeros((M, P), float)
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if Ucg.using_cv and not Ucg.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(self.trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        if len(include_trajs) == 0:
            include_trajs = np.arange(len(self.trajnames))

        n_trajs = len(include_trajs)
        
        N_prev = 0
        count = 0
        for n in range(len(self.trajnames)):
            if n in include_trajs:
                count += 1
            else:
                continue

            if n == len(self.trajnames) - 1:
                print("scalar product traj: {:>5d}/{:<5d} DONE".format(count, n_trajs))
            else:
                print("scalar product traj: {:>5d}/{:<5d}".format(count, n_trajs), end="\r")
            sys.stdout.flush()

            # load eigenvectors
            psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

            if len(cv_names) > 0:
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in cv_names[n] ]).T
            else:
                cv_traj = None

            start_idx = 0
            for chunk in md.iterload(self.trajnames[n], top=self.topfile, chunk=1000):
                N_chunk = chunk.n_frames

                psi_chunk = psi_traj[start_idx:start_idx + N_chunk,:]

                if cv_traj is None:
                    cv_chunk = Ucg.calculate_cv(chunk)
                else:
                    cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_chunk, Ucg.n_dof))

                # calculate gradient of fixed and parametric potential terms
                grad_U1 = Ucg.gradient_U1(xyz_traj, cv_chunk)
                grad_U1 = np.einsum("r,tdr->td", c_r, grad_U1)

                # calculate test function values, gradient, and Laplacian
                test_f = Ucg.test_functions(xyz_traj, cv_chunk)
                grad_f, Lap_f = Ucg.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

                if Ucg.using_U0:
                    grad_U0 = Ucg.gradient_U0(xyz_traj, cv_chunk)
                    curr_psi_gradU0_fj = np.einsum("tm, td,tdp->mp", psi_chunk, -a_coeff*grad_U0, grad_f)
                    psi_gU0_fj = (curr_psi_gradU0_fj + float(N_prev)*psi_gU0_fj)/(float(N_prev + N_chunk))

                # calculate generator terms
                curr_psi_fj = np.einsum("tm,tp->mp", psi_chunk, test_f)
                curr_psi_gradU1_fj = np.einsum("tm,td,tdp->mp", psi_chunk, -a_coeff*grad_U1, grad_f)
                curr_psi_Lap_fj = np.einsum("tm,tp->mp", psi_chunk, (a_coeff/Ucg.beta)*Lap_f)

                psi_fj = (curr_psi_fj + float(N_prev)*psi_fj)/(float(N_prev + N_chunk))
                psi_gU1_fj = (curr_psi_gradU1_fj + float(N_prev)*psi_gU1_fj)/(float(N_prev + N_chunk))
                psi_Lap_fj = (curr_psi_Lap_fj + float(N_prev)*psi_Lap_fj)/(float(N_prev + N_chunk))

                start_idx += N_chunk
                N_prev += N_chunk

        self.psi_fj = psi_fj
        self.psi_gU0_fj = psi_gU0_fj
        self.psi_gU1_fj = psi_gU1_fj
        self.psi_Lap_fj = psi_Lap_fj
        self.psi_Gen_fj = psi_gU0_fj + psi_gU1_fj + psi_Lap_fj 

    def scalar_product_grad_psi_grad_fj(self, Ucg, coeff, psinames, M=1, cv_names=[], include_trajs=[]):

        #TODO

        raise NotImplementedError

        if Ucg.fixed_a_coeff:
            c_r = coeff
            a_coeff = Ucg.a_coeff
        else:
            c_r = coeff[:-1] 
            a_coeff = 1./coeff[-1]  #?

        P = Ucg.n_test_funcs
        if Ucg.constant_a_coeff:
            psi_fj = np.zeros((M, P), float)
            psi_gU0_fj = np.zeros((M, P), float)
            psi_gU1_fj = np.zeros((M, P), float)
            psi_Lap_fj = np.zeros((M, P), float)
            psi_Gen_fj = np.zeros((M, P), float)
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if Ucg.using_cv and not Ucg.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(self.trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        n_trajs = len(self.trajnames) - len(include_trajs)
        N_prev = 0
        count = 0
        for n in range(len(self.trajnames)):
            if n in include_trajs:
                count += 1
            else:
                continue

            if n == len(self.trajnames) - 1:
                print("scalar product traj: {:>5d}/{:<5d} DONE".format(count, n_trajs))
            else:
                print("scalar product traj: {:>5d}/{:<5d}".format(count, n_trajs), end="\r")
            sys.stdout.flush()

            # load eigenvectors
            psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

            if len(cv_names) > 0:
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in cv_names[n] ]).T
            else:
                cv_traj = None

            start_idx = 0
            for chunk in md.iterload(self.trajnames[n], top=self.topfile, chunk=1000):
                N_chunk = chunk.n_frames

                psi_chunk = psi_traj[start_idx:start_idx + N_chunk,:]

                if cv_traj is None:
                    cv_chunk = Ucg.calculate_cv(chunk)
                else:
                    cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_chunk, Ucg.n_dof))

                # calculate gradient of fixed and parametric potential terms
                grad_U1 = Ucg.gradient_U1(xyz_traj, cv_chunk)
                grad_U1 = np.einsum("r,tdr->td", c_r, grad_U1)

                # calculate test function values, gradient, and Laplacian
                test_f = Ucg.test_functions(xyz_traj, cv_chunk)
                grad_f, Lap_f = Ucg.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

                if Ucg.using_U0:
                    grad_U0 = Ucg.gradient_U0(xyz_traj, cv_chunk)
                    curr_psi_gradU0_fj = np.einsum("tm, td,tdp->mp", psi_chunk, -a_coeff*grad_U0, grad_f)
                    psi_gU0_fj = (curr_psi_gradU0_fj + float(N_prev)*psi_gU0_fj)/(float(N_prev + N_chunk))

                # calculate generator terms
                curr_psi_fj = np.einsum("tm,tp->mp", psi_chunk, test_f)
                curr_psi_gradU1_fj = np.einsum("tm,td,tdp->mp", psi_chunk, -a_coeff*grad_U1, grad_f)
                curr_psi_Lap_fj = np.einsum("tm,tp->mp", psi_chunk, (a_coeff/Ucg.beta)*Lap_f)

                psi_fj = (curr_psi_fj + float(N_prev)*psi_fj)/(float(N_prev + N_chunk))
                psi_gU1_fj = (curr_psi_gradU1_fj + float(N_prev)*psi_gU1_fj)/(float(N_prev + N_chunk))
                psi_Lap_fj = (curr_psi_Lap_fj + float(N_prev)*psi_Lap_fj)/(float(N_prev + N_chunk))

                start_idx += N_chunk
                N_prev += N_chunk

        self.psi_fj = psi_fj
        self.psi_gU0_fj = psi_gU0_fj
        self.psi_gU1_fj = psi_gU1_fj
        self.psi_Lap_fj = psi_Lap_fj
        self.psi_Gen_fj = psi_gU0_fj + psi_gU1_fj + psi_Lap_fj 

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
            for chunk in md.iterload(trajnames[n], top=self.topfile, chunk=1000):
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


class LinearForceMatchingLoss(CrossValidatedLoss):

    def __init__(self, topfile, trajnames, savedir, n_cv_sets=5, recalc=False):
        """Creates matrices for minimizing the linear spectral loss equations
        
        Parameters
        ----------
        savedir : str
            Where matrices should be saved.
            
        n_cv_sets : opt.
            Number of k-fold cross validation sets to use.
            
        recalc : bool
            Recalculated 

        """
        CrossValidatedLoss.__init__(self, topfile, trajnames, savedir, n_cv_sets=n_cv_sets)

        self.suffix = "FM"
        self.matrices_estimated = False
        self.recalc = recalc

        if self.matrix_files_exist() and not recalc:
            self._load_matrices()

    @profile
    def calc_matrices(self, Ucg, forcenames, coll_var_names=None, verbose=True, include_trajs=[], chunksize=1000):
        """Calculate eigenpair matrices
       
        Parameters
        ----------
        trajnames : list, str
            Trajectory filenames

        topfile : str
            Filename for topology (pdb)

        forcenames : list, str
            Filenames for 
            
        ti_file : str
            Filename for timescales

        M : int (default=1)
            Number of timescales to keep in eigenfunction expansion.
        
        coll_var_names : list, str (opt)
            Collective variable rilenames if pre-calculated. Will calculate
            collective variables on the fly if not given. 

        """

        self.Ucg = Ucg

        if self.n_cv_sets is None:
            self.n_cv_sets = 1
        else:
            if self.n_cv_sets > 1 and not self.cv_sets_are_assigned:
                self.assign_crossval_sets()

        n_params = Ucg.n_tot_params
        P = Ucg.n_test_funcs

        # if constant diff coeff
        if Ucg.constant_a_coeff:
            d = np.zeros((self.n_cv_sets, P), float)
            X = np.zeros((self.n_cv_sets, P, n_params), float)
            D2 = np.zeros((self.n_cv_sets, n_params, n_params), float)    # TODO: high-dimensional smoothness 
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if Ucg.using_cv and not Ucg.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(forcenames) != len(self.trajnames):
            raise ValueError("Need forces for every trajectory!")

        A_b_set = {}

        #chunksize = 1000
        max_rows = chunksize*Ucg.n_dof

        if len(include_trajs) > 0:
            n_trajs = len(include_trajs)
        else:
            n_trajs = len(self.trajnames)

        count = 0
        N_prev = np.zeros(self.n_cv_sets, float)
        for n in range(len(self.trajnames)):
            if n in include_trajs:
                count += 1
            else:
                continue

            if verbose:
                if count == n_trajs:
                    print("force matching traj: {:>5d}/{:<5d} DONE".format(count, n_trajs))
                else:
                    print("force matching traj: {:>5d}/{:<5d}".format(count, n_trajs), end="\r")
                sys.stdout.flush()

            # load force from simulation 
            force_traj = np.loadtxt(forcenames[n])

            if len(coll_var_names) > 0:
                # load collective variable if given
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in coll_var_names[n] ]).T
            else:
                cv_traj = None

            # calculate matrix for trajectory
            start_idx = 0
            for chunk in md.iterload(self.trajnames[n], top=self.topfile, chunk=chunksize):
                N_chunk = chunk.n_frames
                n_rows = N_chunk*Ucg.n_dof

                f_target_chunk = force_traj[start_idx:start_idx + N_chunk,:Ucg.n_dof]

                if cv_traj is None:
                    cv_chunk = Ucg.calculate_cv(chunk)
                else:
                    cv_chunk = cv_traj[start_idx:start_idx + N_chunk,:]

                # cartesian coordinates unraveled
                xyz_traj = np.reshape(chunk.xyz, (N_chunk, Ucg.n_dof))

                # calculate gradient of fixed and parametric potential terms
                U1_force = -Ucg.gradient_U1(xyz_traj, cv_chunk)

                if Ucg.using_U0:
                    # subtract fixed force from right hand side
                    U0_force = -Ucg.gradient_U0(xyz_traj, cv_chunk)
                    f_target_chunk -= U0_force

                if self.n_cv_sets == 1: 
                    f_cg = np.reshape(U1_force, (n_rows, n_params))
                    f_target = np.reshape(f_target_chunk, (n_rows))

                    if iteration_idx == 0:
                        Q, R = scl.qr(f_cg, mode="economic")

                        A = np.zeros((n_params + max_rows, n_params), float)
                        b = np.zeros(n_params + max_rows, float)

                        A[:n_params,:] = R[:n_params,:].copy()
                        b[:n_params] = np.dot(Q.T, f_target)
                    else:
                        # augment matrix system with next chunk of data
                        A[n_params:n_params + n_rows,:] = f_cg 
                        b[n_params:n_params + n_rows] = f_target

                        Q_next, R_next = scl.qr(A, mode="economic")

                        A[:n_params,:] = R_next
                        b[:n_params] = np.dot(Q_next.T, b)
                    N_prev[0] += float(N_chunk)
                else:
                    for k in range(self.n_cv_sets):   
                        frames_in_this_set = self.cv_set_assignment[n][start_idx:start_idx + N_chunk] == k
                        n_frames_set = np.sum(frames_in_this_set)
                        n_rows_set = n_frames_set*Ucg.n_dof

                        if n_frames_set > 0:
                            f_cg_subset = np.reshape(U1_force[frames_in_this_set], (n_rows_set, n_params))  
                            f_target_subset = np.reshape(f_target_chunk[frames_in_this_set], (n_rows_set))  

                            if not str(k) in A_b_set:
                                Q, R = scl.qr(f_cg_subset, mode="economic")

                                A = np.zeros((n_params + max_rows, n_params), float)
                                b = np.zeros(n_params + max_rows, float)

                                A[:n_params,:] = R[:n_params,:].copy()
                                b[:n_params] = np.dot(Q.T, f_target_subset)

                                A_b_set[str(k)] = (A, b)
                            else:
                                # augment matrix system with next chunk of data
                                (A, b) = A_b_set[str(k)]
                                A[n_params:n_params + n_rows_set,:] = f_cg_subset
                                b[n_params:n_params + n_rows_set] = f_target_subset

                                Q_next, R_next = scl.qr(A, mode="economic")

                                A[:n_params,:] = R_next
                                b[:n_params] = np.dot(Q_next.T, b)

                            N_prev[k] += float(n_frames_set)
                start_idx += N_chunk

        if self.n_cv_sets > 1:
            self.X_sets = [ A_b_set[str(k)][0] for k in range(self.n_cv_sets) ]
            self.d_sets = [ A_b_set[str(k)][1] for k in range(self.n_cv_sets) ]
            self.X = np.sum([ self.set_weights[j]*self.X_sets[j] for j in range(self.n_cv_sets) ], axis=0)
            self.d = np.sum([ self.set_weights[j]*self.d_sets[j] for j in range(self.n_cv_sets) ], axis=0)
        else:
            self.X = A
            self.d = b

        self.matrices_estimated = True
        self._save_matrices()
        self._training_and_validation_matrices()
