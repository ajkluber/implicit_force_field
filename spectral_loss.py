from __future__ import print_function, absolute_import
import os
import sys
import numpy as np

from scipy.optimize import least_squares
from scipy.optimize import minimize
import scipy.linalg

import mdtraj as md

# TODO: nonlinear loss function

class CrossValidatedLoss(object):
    def __init__(self, savedir, n_cv_sets=5):
        """Creates matrices for minimizing the linear spectral loss equations
        
        Parameters
        ----------
        savedir : str
            Where matrices should be saved.
            
        n_cv_sets : opt.
            Number of k-fold cross validation sets to use.
            
        """

        self.savedir = savedir
        self.n_cv_sets = n_cv_sets
        self.cv_sets_are_assigned = False

    def assign_crossval_sets(self, topfile, trajnames):
        """Randomly assign frames to training and validation sets

        Parameters
        ----------
        topfile : str
            Topology filename.

        trajnames : list, str
            List of trajectory filenames.

        """

        self.topfile = topfile
        self.trajnames = trajnames

        set_assignment = []
        traj_n_frames = []
        for n in range(len(trajnames)):
            length = 0
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
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

class LinearLoss(CrossValidatedLoss):

    def __init__(self, savedir, n_cv_sets=5, recalc=False):
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
        CrossValidatedLoss.__init__(self, savedir, n_cv_sets=n_cv_sets)

        self.matrices_estimated = False
        self.recalc = recalc

        if self.matrix_files_exist() and not recalc:
            self._load_matrices()

    def calc_matrices(self, Ucg, topfile, trajnames, psinames, ti_file, M=1, coll_var_names=None, verbose=True):
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
                self.assign_crossval_sets(topfile, trajnames)

        R = Ucg.n_params
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

        if len(psinames) != len(trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        kappa = 1./np.load(ti_file)[:M]

        N_prev = np.zeros(self.n_cv_sets, float)
        for n in range(len(trajnames)):
            if verbose:
                if n == len(trajnames) - 1:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d} DONE".format(n + 1, len(trajnames)))
                else:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d}".format(n + 1, len(trajnames)), end="\r")
                sys.stdout.flush()

            # load eigenvectors
            psi_traj = np.array([ np.load(temp_psiname) for temp_psiname in psinames[n] ]).T

            if len(coll_var_names) > 0:
                # load collective variable if given
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in coll_var_names[n] ]).T
            else:
                cv_traj = None

            # calculate matrix for trajectory
            #matrix_elements_for_traj(trajnames[n], topfile, N_prev, )

            start_idx = 0
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
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

    def matrix_files_exist(self):
        X_files_exist = np.all([ os.path.exists("{}/X_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ])
        d_files_exist = np.all([ os.path.exists("{}/d_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ])
        set_files_exist = np.all([ os.path.exists("{}/frame_set_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ])
        files_exist = X_files_exist and d_files_exist and set_files_exist

        return files_exist

    def _save_matrices(self): 
        for k in range(self.n_cv_sets):
            np.save("{}/X_{}.npy".format(self.savedir, k + 1), self.X_sets[k])
            np.save("{}/d_{}.npy".format(self.savedir, k + 1), self.d_sets[k])
            np.save("{}/frame_set_{}.npy".format(self.savedir,  k + 1), self.cv_set_assignment[k])    

        np.save("{}/X.npy".format(self.savedir), self.X)
        np.save("{}/d.npy".format(self.savedir), self.d)

    def _load_matrices(self):
        
        if self.n_cv_sets is None:
            raise ValueErro("Need to define number of cross val sets in order to load them")

        print("Loaded saved matrices...")
        self.X_sets = [ np.load("{}/X_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ]
        self.d_sets = [ np.load("{}/d_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ]
        set_assignment = [ np.load("{}/frame_set_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ]

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
                X_train, X_val = self.X_train_val[k]
                d_train, d_val = self.d_train_val[k]

                X_reg = np.dot(X_train.T, X_train) + alphas[i]*D
                d_reg = np.dot(X_train.T, d_train)
                coeff_fold = scipy.linalg.lstsq(X_reg, d_reg, cond=1e-10)[0]
        
                train_mse_folds.append(np.mean((np.dot(X_train, coeff_fold) - d_train)**2))
                valid_mse_folds.append(np.mean((np.dot(X_val, coeff_fold) - d_val)**2))
        
            train_mse.append([np.mean(train_mse_folds), np.std(train_mse_folds)/np.sqrt(float(self.n_cv_sets))])
            valid_mse.append([np.mean(valid_mse_folds), np.std(valid_mse_folds)/np.sqrt(float(self.n_cv_sets))])

            X_reg = np.dot(self.X.T, self.X) + alphas[i]*D
            d_reg = np.dot(self.X.T, self.d)
            coeffs.append(scipy.linalg.lstsq(X_reg, d_reg, cond=1e-10)[0])

        self.alphas = alphas
        self.coeffs = np.array(coeffs)
        self.train_mse = np.array(train_mse)
        self.valid_mse = np.array(valid_mse)

