from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
import scipy.linalg as scl

from scipy.optimize import least_squares
from scipy.optimize import minimize
import scipy.optimize
import scipy.linalg

import mdtraj as md


#from memory_profiler import profile
# TODO: nonlinear loss function

class CrossValidatedLoss(object):
    def __init__(self, topfile, trajnames, savedir, n_cv_sets=5,
            save_by_traj=False, recalc=False):
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
        self.recalc = recalc
        self.save_by_traj = save_by_traj

    def assign_crossval_sets(self):
        """Randomly assign frames to training and validation sets

        Parameters
        ----------
        topfile : str
            Topology filename.

        trajnames : list, str
            List of trajectory filenames.

        """

        self.cv_set_assignment = []
        traj_n_frames = []
        for n in range(len(self.trajnames)):
            length = 0
            for chunk in md.iterload(self.trajnames[n], top=self.topfile, chunk=1000):
                length += chunk.n_frames
            traj_n_frames.append(length)

            self.cv_set_assignment.append(np.random.randint(low=0, high=self.n_cv_sets, size=length))
        self.total_n_frames = sum(traj_n_frames)

        self.n_frames_in_set = []
        for k in range(self.n_cv_sets):
            self.n_frames_in_set.append(np.sum([ np.sum(self.cv_set_assignment[i] == k) for i in range(len(self.cv_set_assignment)) ]))
        self.set_weights = [ (self.n_frames_in_set[j]/float(self.total_n_frames)) for j in range(self.n_cv_sets) ]

        self.cv_sets_are_assigned = True

    def Xname_by_traj(self, run_idx, traj_idx, cv_set_idx):
        return "{}/run_{}_{}_X_{}_{}.npy".format(self.savedir, run_idx, traj_idx, self.suffix, cv_set_idx)

    def dname_by_traj(self, run_idx, traj_idx, cv_set_idx):
        return "{}/run_{}_{}_d_{}_{}.npy".format(self.savedir, run_idx, traj_idx, self.suffix, cv_set_idx)

    def frame_set_name_by_traj(self, run_idx, traj_idx):
        return "{}/run_{}_{}_frame_set_{}.npy".format(self.savedir, run_idx, traj_idx, self.suffix)

    def Xname(self, cv_set_idx):
        return "{}/X_{}_{}.npy".format(self.savedir, self.suffix, cv_set_idx)

    def dname(self, cv_set_idx):
        return "{}/d_{}_{}.npy".format(self.savedir, self.suffix, cv_set_idx)

    def frame_set_name(self, traj_idx):
        return "{}/frame_set_{}_{}.npy".format(self.savedir, self.suffix, traj_idx)

    def matrix_files_exist(self):

        if self.save_by_traj:
            X_files_exist = []
            d_files_exist = []
            set_files_exist = []
            for i in range(len(self.trajnames)):
                tname = trajnames[i]
                idx1 = (os.path.dirname(tname)).split("_")[-1]
                idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
                for n in range(self.n_cv_sets):
                    X_files_exist.append(os.path.exists(self.Xname_by_traj(idx1, idx2, n + 1)))
                    d_files_exist.append(os.path.exists(self.dname_by_traj(idx1, idx2, n + 1)))

                set_files_exist.append(os.path.exists(self.frame_set_name_by_traj(idx1, idx2)))
            files_exist = np.all(X_files_exist) and np.all(d_files_exist) and np.all(set_files_exist)
        else:
            X_files_exist = np.all([ os.path.exists(self.Xname(i + 1)) for i in range(self.n_cv_sets) ])
            d_files_exist = np.all([ os.path.exists(self.dname(i + 1)) for i in range(self.n_cv_sets) ])
            set_files_exist = np.all([ os.path.exists(self.frame_set_name(i + 1)) for i in range(len(self.trajnames)) ])
            files_exist = X_files_exist and d_files_exist and set_files_exist

        return files_exist

    def _save_matrices(self): 
        for k in range(self.n_cv_sets):
            np.save(self.Xname(k + 1), self.X_sets[k])
            np.save(self.dname(k + 1), self.d_sets[k])

        for k in range(len(self.trajnames)):
            np.save(self.frame_set_name(k + 1), self.cv_set_assignment[k])    

        np.save("{}/X_{}.npy".format(self.savedir, self.suffix), self.X)
        np.save("{}/d_{}.npy".format(self.savedir, self.suffix), self.d)


    def _load_matrices(self):
        
        print("Loading saved matrices...")
        if self.n_cv_sets is None:
            raise ValueErro("Need to define number of cross val sets in order to load them")

        self.cv_set_assignment = []

        if self.save_by_traj:
            self.X_sets = [ [] for i in range(n_cv_sets) ]
            self.d_sets = [ [] for i in range(n_cv_sets) ]

            if self.suffix == "FM":
                # force-matching is done by QR decomposition so matrices for 
                # each trajectory need to be concatenated
                for i in range(len(self.trajnames)):
                    tname = self.trajnames[i]
                    idx1 = (os.path.dirname(tname)).split("_")[-1]
                    idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]

                    self.cv_set_assignment.append(np.load(self.frame_set_name_by_traj(idx1, idx2)))
                    for n in range(self.n_cv_sets):
                        Xtemp = np.load(self.Xname_by_traj(idx1, idx2, n + 1))
                        dtemp = np.load(self.dname_by_traj(idx1, idx2, n + 1))

                        if i == 0:
                            self.X_sets[n] = Xtemp
                            self.d_sets[n] = dtemp
                        else:
                            self.X_sets[n] = np.concatenate([X_sets[n], Xtemp], axis=0)
                            self.d_sets[n] = np.concatenate([d_sets[n], dtemp], axis=0)

            elif self.suffix == "EG":
                # eigenpair matrices for each traj can be summed together with
                # weights
                X_by_traj = [ [] for i in range(self.n_cv_sets) ]
                d_by_traj = [ [] for i in range(self.n_cv_sets) ]

                cv_frames_by_traj = [ [] for i in range(self.n_cv_sets) ]
                cv_frames_by_traj = np.zeros((len(self.trajnames), self.n_cv_sets), float)
                for i in range(len(self.trajnames)):
                    tname = trajnames[i]
                    idx1 = (os.path.dirname(tname)).split("_")[-1]
                    idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]

                    frm_traj = np.load(self.frame_set_name_by_traj(idx1, idx2))

                    for n in range(self.n_cv_sets): 
                        cv_frames_by_traj[i, n].append(np.sum(frm_traj == n))

                    self.cv_set_assignment.append(frm_traj)

                self.n_frames_in_set = np.sum(cv_frames_by_traj, axis=0) 

                #TODO: check this works

                for i in range(len(self.trajnames)):
                    tname = trajnames[i]
                    idx1 = (os.path.dirname(tname)).split("_")[-1]
                    idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]

                    for n in range(self.n_cv_sets):
                        Xtemp = np.load(self.Xname_by_traj(idx1, idx2, n + 1))
                        dtemp = np.load(self.dname_by_traj(idx1, idx2, n + 1))

                        if i == 0:
                            self.X_sets[n] = Xtemp
                            self.d_sets[n] = dtemp
                        else:
                            w_for_set = cv_frames_by_traj[i, n]/float(self.n_frames_in_set[n])
                            self.X_sets[n] += w_for_set*Xtemp
                            self.d_sets[n] += w_for_set*dtemp

                self.total_n_frames = np.sum(self.n_frames_in_set)
                self.set_weights = [ self.n_frames_in_set[n]/float(self.total_n_frames) for n in range(self.n_cv_sets) ] 

        else:
            self.X_sets = []
            self.d_sets = []
            for n in range(self.n_cv_sets):
               self.X_sets.append(np.load(self.Xname(n + 1)))
               self.d_sets.append(np.load(self.dname(n + 1)))

            for i in range(len(self.trajnames)):
                self.cv_set_assignment.append(np.load(self.frame_set_name(i + 1)))

            self.n_frames_in_set = []
            for k in range(self.n_cv_sets):
                self.n_frames_in_set.append(np.sum([ np.sum(self.cv_set_assignment[i] == k) for i in range(len(self.trajnames)) ]))
            self.total_n_frames = np.sum(self.n_frames_in_set)
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

            if self.save_by_traj and self.suffix == "FM":
                # force-matching matrices are created by QR decomp. Should be
                # concatenated not averaged.
                train_X = []
                train_d = []
                for j in range(self.n_cv_sets):
                    if j != i:
                        train_X.append(self.X_sets[j])
                        train_d.append(self.d_sets[j])
                train_X = np.concatenate(train_X, axis=0)
                train_d = np.concatenate(train_d, axis=0)
            else:
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

class OneDimSpectralLoss(object):

    def __init__(self, model, kappa, psi_trajs, x_trajs, savedir=".",
            n_cv_sets=5, recalc=False, suffix="EG1d", softplus_coeff_a=False):
        # indices of coefficients that are to remain positive. Will apply
        # softplus functions

        self.savedir = savedir
        self.n_cv_sets = n_cv_sets
        self.cv_sets_are_assigned = False
        self.recalc = recalc
        self.softplus_coeff_a = softplus_coeff_a  
        self.suffix = suffix

        self.model = model
        self.const_a = model.const_a
        self.fixed_a = model.fixed_a
        self.beta_kappa = model.beta*kappa


        self.psi_trajs = psi_trajs
        self.x_trajs = x_trajs
        self.n_trajs = len(x_trajs)
        self.P = model.n_test_funcs
        self.R_U = model.n_pot_params

        if self.fixed_a:
            self.const_a = True

        if self.fixed_a:
            self.a_c = model.a_c
            self.R_a = 0
        else:
            if self.const_a:
                self.eval_loss = self._eval_loss_const_a
                self.eval_grad_loss = self._eval_grad_loss_const_a
                #self.eval_hess_loss = self._eval_hess_loss_const_a
                self.R_a = 1
            else:
                self.calc_matrices = self._matrix_elements_general_a
                self.eval_loss_reg = self._eval_loss_general_a_with_regularization
                self.eval_loss = self._eval_loss_general_a
                self.eval_grad_loss = self._eval_grad_loss_general_a
                self.eval_hess_loss = self._eval_hess_loss_general_a
                self.R_a = model.n_noise_params

        self.R = self.R_U + self.R_a

        self.matrices_estimated = False
        if self.matrix_files_exist() and not self.recalc:
            print("loading matrices...")
            self._load_matrices()
        else:
            self.assign_crossval_sets()
            self.calc_matrices()

    def assign_crossval_sets(self):
        """Randomly assign frames to training and validation sets"""

        self.cv_set_assignment = []
        traj_n_frames = []
        for n in range(len(self.x_trajs)):
            length = self.x_trajs[n].shape[0]
            traj_n_frames.append(length)
            self.cv_set_assignment.append(np.random.randint(low=0, high=self.n_cv_sets, size=length))
        self.total_n_frames = sum(traj_n_frames)

        self.n_frames_in_set = []
        for k in range(self.n_cv_sets):
            self.n_frames_in_set.append(np.sum([ np.sum(self.cv_set_assignment[i] == k) for i in range(len(self.cv_set_assignment)) ]))
        self.set_weights = np.array([ (self.n_frames_in_set[j]/float(self.total_n_frames)) for j in range(self.n_cv_sets) ])

        self.weight_without_k = np.zeros((self.n_cv_sets, self.n_cv_sets))
        for i in range(self.n_cv_sets):
            n_frm_i = self.n_frames_in_set[i]
            for j in range(self.n_cv_sets):
                if j != i:
                    self.weight_without_k[i,j] = self.n_frames_in_set[j]/float(self.total_n_frames - n_frm_i)

        self.cv_sets_are_assigned = True

    def matrix_files_exist(self):
        """Check if files exist"""
        fnames = ["psi_a_dU_df", "psi_da_df", "psi_a_d2f", "psi_f"]
        files_exist = np.all([ os.path.exists("{}/{}_{}.npy".format(self.savedir, fname, self.suffix)) for fname in fnames ])
        return files_exist

    def _load_matrices(self):
        """Load matrix elements from file"""

        psi_a_dU_df = np.load("{}/psi_a_dU_df_{}.npy".format(self.savedir, self.suffix))
        psi_da_df = np.load("{}/psi_da_df_{}.npy".format(self.savedir, self.suffix))
        psi_a_d2f = np.load("{}/psi_a_d2f_{}.npy".format(self.savedir, self.suffix))
        psi_f = np.load("{}/psi_f_{}.npy".format(self.savedir, self.suffix))

        if self.const_a:
            # only keep constant noise terms
            self.psi_a_dU_df = psi_a_dU_df[:,0,:,:]
            self.psi_da_df = np.zeros(psi_da_df.shape)
            self.psi_a_d2f = psi_a_d2f[:,0,:] 
            self.psi_f = psi_f
        else:
            self.psi_a_dU_df = psi_a_dU_df
            self.psi_da_df = psi_da_df
            self.psi_a_d2f = psi_a_d2f
            self.psi_f = psi_f

        # load assignments for cross validation folds/sets
        self.cv_set_assignment = []
        traj_n_frames = []
        for i in range(len(self.x_trajs)):
            frm_set = np.load("{}/frame_set_{}_{}.npy".format(self.savedir, self.suffix, i + 1))
            self.cv_set_assignment.append(frm_set)
            traj_n_frames.append(frm_set.shape[0])

        self.total_n_frames = sum(traj_n_frames)

        self.n_frames_in_set = []
        for k in range(self.n_cv_sets):
            self.n_frames_in_set.append(np.sum([ np.sum(self.cv_set_assignment[i] == k) for i in range(len(self.cv_set_assignment)) ]))
        self.set_weights = np.array([ (self.n_frames_in_set[j]/float(self.total_n_frames)) for j in range(self.n_cv_sets) ])

        self.weight_without_k = np.zeros((self.n_cv_sets, self.n_cv_sets))
        for i in range(self.n_cv_sets):
            n_frm_i = self.n_frames_in_set[i]
            for j in range(self.n_cv_sets):
                if j != i:
                    self.weight_without_k[i,j] = self.n_frames_in_set[j]/float(self.total_n_frames - n_frm_i)

        self.cv_sets_are_assigned = True
        self.matrices_estimated = True

    def _save_matrices(self):

        np.save("{}/psi_a_dU_df_{}.npy".format(self.savedir, self.suffix), self.psi_a_dU_df)
        np.save("{}/psi_da_df_{}.npy".format(self.savedir, self.suffix), self.psi_da_df)
        np.save("{}/psi_a_d2f_{}.npy".format(self.savedir, self.suffix), self.psi_a_d2f)
        np.save("{}/psi_f_{}.npy".format(self.savedir, self.suffix), self.psi_f)

        for i in range(len(self.cv_set_assignment)):
            np.save("{}/frame_set_{}_{}.npy".format(self.savedir, self.suffix, i + 1), self.cv_set_assignment[i])

    def _matrix_elements_general_a(self):
        """Calculate spectral loss matrix elements over trajectories"""

        psi_a_dU_df = np.zeros((self.n_cv_sets, self.R_a, self.R_U, self.P), float)
        psi_da_df = np.zeros((self.n_cv_sets, self.R_a, self.P), float)
        psi_a_d2f = np.zeros((self.n_cv_sets, self.R_a, self.P), float)
        psi_f = np.zeros((self.n_cv_sets, self.P), float)

        N_prev = np.zeros(self.n_cv_sets, float)

        for i in range(self.n_trajs):
            print("calculating matrix elements ({}/{})".format(i + 1, self.n_trajs), end="\r")
            sys.stdout.flush()
            x = self.x_trajs[i]
            psi = self.psi_trajs[i]
            N_curr = x.shape[0]
            
            # evaluate force, noise, and test functions on trajectory
            a_basis, da_basis = self.model.eval_a_basis(x)
            dU1_basis = self.model.eval_dU1_basis(x)
            f, df, d2f = self.model.eval_f(x)

            # average over different cross validation sets
            for k in range(self.n_cv_sets):
                frames_in_this_set = self.cv_set_assignment[i] == k
                N_curr_set = np.sum(frames_in_this_set)

                if N_curr_set > 0:
                    psi_subset = psi[frames_in_this_set]
                    a_subset = a_basis[frames_in_this_set]
                    da_subset = da_basis[frames_in_this_set]
                    dU1_subset = -dU1_basis[frames_in_this_set]

                    f_subset = f[frames_in_this_set]
                    df_subset = df[frames_in_this_set]
                    d2f_subset = d2f[frames_in_this_set]

                    # inner products with eigenvector
                    curr_psi_a_dU_df = np.einsum("tn,tm,tp->nmp", a_subset, dU1_subset, df_subset)
                    curr_psi_da_df = np.einsum("tn,tp->np", da_subset, df_subset)
                    curr_psi_a_d2f = np.einsum("tn,tp->np", a_subset, d2f_subset)
                    curr_psi_f = np.einsum("t,tp->p", psi_subset, f_subset)

                    # running average
                    psi_a_dU_df[k] = (curr_psi_a_dU_df + N_prev[k]*psi_a_dU_df[k])/float(N_prev[k] + N_curr_set)
                    psi_da_df[k] = (curr_psi_da_df + N_prev[k]*psi_da_df[k])/float(N_prev[k] + N_curr_set)
                    psi_a_d2f[k] = (curr_psi_a_d2f + N_prev[k]*psi_a_d2f[k])/float(N_prev[k] + N_curr_set)
                    psi_f[k] = (curr_psi_f + N_prev[k]*psi_f[k])/float(N_prev[k] + N_curr_set)

                    N_prev[k] += N_curr_set

        print("calculating matrix elements ({}/{}) DONE".format(i + 1, self.n_trajs))
        sys.stdout.flush()
        self.psi_a_dU_df = psi_a_dU_df
        self.psi_da_df = psi_da_df
        self.psi_a_d2f = psi_a_d2f
        self.psi_f = self.beta_kappa*psi_f

        self.matrices_estimated = True
        self._save_matrices()

    ########################################################################
    # MINIMIZE LOSS
    ########################################################################
    def _solve_fixed_a(self, a_c, alphas):
        """Solve for potential parameters when diffusion cefficient is fixed"""

        X = np.einsum("w,wmp->mp", self.set_weights, self.psi_a_dU_df) 
        d = -np.einsum("w,wp->p", self.set_weights, self.psi_f)/a_c
        d += -np.einsum("w,wp->p", self.set_weights, self.psi_a_d2f)

        return np.linalg.lstsq(X.T, d, rcond=1e-10)[0]

    def _solve_fixed_a(self, a_c):
        """Solve for potential parameters when diffusion cefficient is fixed"""

        X = np.einsum("w,wmp->mp", self.set_weights, self.psi_a_dU_df) 
        d = -np.einsum("w,wp->p", self.set_weights, self.psi_f)/a_c
        d += -np.einsum("w,wp->p", self.set_weights, self.psi_a_d2f)

        return np.linalg.lstsq(X.T, d, rcond=1e-10)[0]
        #return scipy.optimize.lsq_linear(X, d)

    def _solve_const_a_linear(self, alphas):
        """Solve for potential parameters when diffusion cefficient is fixed"""
        sqrt_k = np.sqrt(float(self.n_cv_sets))

        X_train_val = []
        d_train_val = []
        for k in range(self.n_cv_sets):
            # const diff
            train_X_1 = np.einsum("w,wmp->mp", self.weight_without_k[k], self.psi_a_dU_df) 
            train_X_2 = np.einsum("w,wp->p", self.weight_without_k[k], self.psi_f) 
            train_X = np.concatenate([train_X_1, train_X_2[np.newaxis,:]])
            train_d = -np.einsum("w,wp->p", self.weight_without_k[k], self.psi_a_d2f)

            valid_X = np.concatenate([ self.psi_a_dU_df[k,:,:], (self.psi_f[k])[np.newaxis,:]], axis=0)
            valid_d = -self.psi_a_d2f[k]

            X_train_val.append([ train_X, valid_X ])
            d_train_val.append([ np.dot(train_X, train_d), valid_d ])

        X_1 = np.einsum("w,wmp->mp", self.set_weights, self.psi_a_dU_df) 
        X_2 = np.einsum("w,wp->p", self.set_weights, self.psi_f)
        X = np.concatenate([X_1, X_2[np.newaxis,:]], axis=0)
        d = -np.einsum("w,wp->p", self.set_weights, self.psi_a_d2f)
        d_reg = np.dot(X, d)

        I = np.identity(X.shape[0])

        all_coeffs = []
        all_train_mse = []
        all_valid_mse = []
        for i in range(len(alphas)):
            print("{}/{}".format(i + 1, len(alphas)), end="\r")
            sys.stdout.flush()
            alpha = alphas[i]

            trn_mse_folds = []
            val_mse_folds = []
            for k in range(self.n_cv_sets):
                X_train, X_valid = X_train_val[k]
                d_train_reg, d_valid = d_train_val[k]

                X_reg_k = np.dot(X_train, X_train.T) + alpha*I
                coeff = np.linalg.lstsq(X_reg_k, d_train_reg, rcond=1e-10)[0]
                
                train_mse = np.mean((np.dot(X_reg_k, coeff) - d_train_reg)**2)
                valid_mse = np.mean((np.dot(X_valid.T, coeff) - d_valid)**2)

                trn_mse_folds.append(train_mse)
                val_mse_folds.append(valid_mse)

            X_reg = np.dot(X, X.T) + alpha*I
            all_coeffs.append(np.linalg.lstsq(X_reg, d_reg, rcond=1e-10)[0])

            all_train_mse.append([ np.mean(trn_mse_folds), np.std(trn_mse_folds)/sqrt_k ])
            all_valid_mse.append([ np.mean(val_mse_folds), np.std(val_mse_folds)/sqrt_k ])
        print("{}/{} DONE".format(i + 1, len(alphas)), end="\r")
        sys.stdout.flush()

        return np.array(all_coeffs), np.array(all_train_mse), np.array(all_valid_mse)

    def solve(self, coeff0, rdg_alpha_U, rdg_alpha_a, method="CG"):
        """Solve for coefficients and calculate cross-validation score"""
        sqrt_k = np.sqrt(float(self.n_cv_sets))

        all_coeffs = []
        all_avg_cv_scores = []
        all_std_cv_scores = []
        print("solving with cross-validation...")
        for i in range(len(rdg_alpha_U)):
            alpha_U = rdg_alpha_U[i]
            temp_coeffs = []
            temp_avg_cv = []
            temp_std_cv = []

            for j in range(len(rdg_alpha_a)):
                alpha_a = rdg_alpha_a[j]

                print("  {:>4d}/{:<4d}    {:>4d}/{:<4d}".format(i + 1, len(rdg_alpha_U), j + 1, len(rdg_alpha_a)), end="\r")
                sys.stdout.flush()

                # for each value of the reg params have the mean and std dev of loss over
                avg_cv = np.zeros(self.n_cv_sets)
                for k in range(self.n_cv_sets):
                    opt_soln = minimize(self.eval_loss_reg, coeff0, method=method, args=(k, alpha_U, alpha_a))

                    # cross validation score across other test sets
                    cv_k = 0
                    for kprime in range(self.n_cv_sets):
                        if kprime != k:
                            loss_k = self.eval_loss(opt_soln.x, kprime) 
                            
                            cv_k += self.weight_without_k[k, kprime]*loss_k
                    avg_cv[k] = cv_k
                    #coeff0 = np.copy(opt_soln.x)

                temp_coeffs.append(opt_soln.x)
                temp_avg_cv.append(np.mean(avg_cv))
                temp_std_cv.append(np.std(avg_cv)/sqrt_k)

            all_coeffs.append(temp_coeffs)
            all_avg_cv_scores.append(temp_avg_cv)
            all_std_cv_scores.append(temp_std_cv)

        print("  {:>4d}/{:<4d}    {:>4d}/{:<4d}  DONE".format(i + 1, len(rdg_alpha_U), j + 1, len(rdg_alpha_a)))
        sys.stdout.flush()

        return np.array(all_coeffs), np.array(all_avg_cv_scores), np.array(all_std_cv_scores)

    def _solve_general_a(self, coeff0, alphas, n_iters=10):

        alpha_U, alpha_a = alphas
        I_U = np.identity(self.R_U)
        I_a = np.identity(self.R_a)

        prev_cU = coeff0[:self.R_U]
        prev_ca = coeff0[self.R_U:]

        print("solving...")
        diff_cU = []
        diff_ca = []
        for i in range(n_iters):
            # for many iterations solve
            print(" {}/{}".format(i + 1, n_iters), end="\r")
            A_U = np.einsum("n,nmp->mp", prev_ca, self.psi_a_dU_df).T
            b_U = -(np.einsum("n,np->p", prev_ca, self.psi_da_df + self.psi_a_d2f) + self.psi_f)

            Areg_U = np.dot(A_U.T, A_U) + alpha_U*I_U
            breg_U = np.dot(A_U.T, b_U)
            new_cU = np.linalg.lstsq(Areg_U, breg_U)[0]

            A_a = (np.einsum("nmp,m->np", self.psi_a_dU_df, new_cU) + self.psi_da_df + self.psi_a_d2f).T
            b_a = -self.psi_f

            Areg_a = np.dot(A_a.T, A_a) + alpha_a*I_a
            breg_a = np.dot(A_a.T, b_a)
            new_ca = np.linalg.lstsq(Areg_a, breg_a)[0]

            diff_cU.append(np.sum((new_cU - prev_cU)**2))
            diff_ca.append(np.sum((new_ca - prev_ca)**2))

            prev_cU = np.copy(new_cU)
            prev_ca = np.copy(new_ca)

        coeff = np.concatenate([new_cU, new_ca])
        # calculate cross-validation score with final coefficients
        print(" {}/{} DONE".format(i + 1, n_iters))

        return coeff, diff_cU, diff_ca

    def _solve_loss_const_a(self, coeff, k):
        # Loss = (1/2)sum_j (<psi, Lf> + beta*kappa*<psi, f>)^2
        # where
        #       <psi, Lf>  = <psi, (-a dF + da)*df + a*d2f >

        # Hess_Loss = sum_j (<psi, Lf> + beta*kappa*<psi, f>)*<psi, d2Lf_dck_dck'> + <psi, dLf_dck>*<psi, dLf_dck'>

        coeff_U = coeff[:-1]
        if self.softplus_coeff_a:
            a_c = np.log(1 + np.exp(coeff[-1]))
        else:
            a_c = coeff[-1]

        if k == -1:
            # loss for all data
            Sum_j = np.einsum("w,wmp,m->p", self.set_weights, a_c*self.psi_a_dU_df, coeff_U) 
            Sum_j += np.einsum("w,wp->p", self.set_weights, a_c*self.psi_a_d2f)
            Sum_j += np.einsum("w,wp->p", self.set_weights, self.psi_f)
        else:
            # loss for set k
            Sum_j = np.einsum("mp,m->p", a_c*self.psi_a_dU_df[k], coeff_U) 
            Sum_j += a_c*self.psi_a_d2f[k]
            Sum_j += self.psi_f[k]

        return 0.5*np.sum(Sum_j**2)

    def _eval_loss_const_a(self, coeff, k):
        # Loss = (1/2)sum_j (<psi, Lf> + beta*kappa*<psi, f>)^2
        # where
        #       <psi, Lf>  = <psi, (-a dF + da)*df + a*d2f >

        # Hess_Loss = sum_j (<psi, Lf> + beta*kappa*<psi, f>)*<psi, d2Lf_dck_dck'> + <psi, dLf_dck>*<psi, dLf_dck'>

        coeff_U = coeff[:-1]
        if self.softplus_coeff_a:
            a_c = np.log(1 + np.exp(coeff[-1]))
        else:
            a_c = coeff[-1]

        if k == -1:
            # loss for all data
            Sum_j = np.einsum("w,wmp,m->p", self.set_weights, a_c*self.psi_a_dU_df, coeff_U) 
            Sum_j += np.einsum("w,wp->p", self.set_weights, a_c*self.psi_a_d2f)
            Sum_j += np.einsum("w,wp->p", self.set_weights, self.psi_f)
        else:
            # loss for set k
            Sum_j = np.einsum("mp,m->p", a_c*self.psi_a_dU_df[k], coeff_U) 
            Sum_j += a_c*self.psi_a_d2f[k]
            Sum_j += self.psi_f[k]

        return 0.5*np.sum(Sum_j**2)

    def _eval_grad_loss_const_a(self, coeff, dummy):
        # Grad_Loss = sum_j (<psi, Lf> + beta*kappa*<psi, f>)*<psi, dLf_dck>

        coeff_U = coeff[:-1]
        if self.softplus_coeff_a:
            a_c = np.log(1 + np.exp(coeff[-1]))
            grad_a_c = np.exp(coeff[-1])/(1 + np.exp(coeff[-1]))
        else:
            a_c = coeff[-1]

        Sum_j = np.einsum("w,wmp,m->p", self.set_weights, a_c*self.psi_a_dU_df, coeff_U) 
        Sum_j += np.einsum("w,wp->p", self.set_weights, a_c*self.psi_a_d2f)
        Sum_j += np.einsum("w,wp->p", self.set_weights, self.psi_f)

        grad_sum_j_coeff_U = np.einsum("w,wmp->mp", self.set_weights, a_c*self.psi_a_dU_df) 

        grad_sum_j_coeff_a = np.einsum("w,wmp,m->p", self.set_weights, self.psi_a_dU_df, coeff_U) 
        grad_sum_j_coeff_a += np.einsum("w,wp->p", self.set_weights, self.psi_a_d2f)

        if self.softplus_coeff_a:
            grad_coeff_a = np.array([grad_a_c*np.sum(Sum_j*grad_sum_j_coeff_a)])
        else:
            grad_coeff_a = np.array([np.sum(Sum_j*grad_sum_j_coeff_a)])

        grad_coeff_U = np.einsum("p,mp->m", Sum_j, grad_sum_j_coeff_U)
        grad_loss = np.concatenate([grad_coeff_U, grad_coeff_a])

        return grad_loss

    def _eval_hess_loss_const_a(self, coeff, dummy):
        raise NotImplementedError

    def _eval_loss_general_a(self, coeff, k):
        """Loss function for variable diffusion function
        
        Enforces that diffusion function is positive"""

        coeff_U = coeff[:self.R_U]
        if self.softplus_coeff_a:
            coeff_a = np.log(1 + np.exp(np.copy(coeff[self.R_U:])))

        if k == -1:
            # loss for all data
            Sum_j = np.einsum("w,n,wnmp,m->p", self.set_weights, coeff_a, self.psi_a_dU_df, coeff_U) 
            Sum_j += np.einsum("w,n,wnp->p", self.set_weights, coeff_a, self.psi_da_df)
            Sum_j += np.einsum("w,n,wnp->p", self.set_weights, coeff_a, self.psi_a_d2f)
            Sum_j += np.einsum("w,wp->p", self.set_weights, self.psi_f)
        else:
            # loss for set k
            Sum_j = np.einsum("n,nmp,m->p", coeff_a, self.psi_a_dU_df[k], coeff_U) 
            Sum_j += np.einsum("n,np->p",  coeff_a, self.psi_da_df[k])
            Sum_j += np.einsum("n,np->p", coeff_a, self.psi_a_d2f[k])
            Sum_j += self.psi_f[k]

        return 0.5*np.sum(Sum_j**2)

    def _eval_loss_general_a_with_regularization(self, coeff, k, alpha_U, alpha_a):
        """Loss function for variable diffusion function
        
        Enforces that diffusion function is positive"""

        pos_ca = np.log(1 + np.exp(coeff[self.R_U:]))
        spectral_loss = self._eval_loss_general_a(coeff, k)
        ridge_loss = alpha_U*np.sum(coeff[:self.R_U]**2) + alpha_a*np.sum(pos_ca**2)

        return spectral_loss + ridge_loss

    #def _eval_grad_loss_general_a_coeff_a(self, coeff_a, coeff_U):

    def _eval_grad_loss_general_a(self, coeff, dummy):

        coeff_U = coeff[:self.R_U]
        if self.softplus_coeff_a:
            coeff_a = np.log(1 + np.exp(np.copy(coeff[self.R_U:])))
            grad_coeff_a = np.exp(coeff[self.R_U:])/(1 + np.exp(coeff[self.R_U:]))
        else:
            coeff_a = coeff[self.R_U:]
            
        Sum_j = np.einsum("w,n,wnmp,m->p", self.set_weights, coeff_a, self.psi_a_dU_df, coeff_U) 
        Sum_j += np.einsum("w,n,wnp->p", self.set_weights, coeff_a, self.psi_da_df)
        Sum_j += np.einsum("w,n,wnp->p", self.set_weights, coeff_a, self.psi_a_d2f)
        Sum_j += np.einsum("w,wp->p", self.set_weights, self.psi_f)

        grad_sum_j_coeff_U = np.einsum("w,n,wnmp->mp", self.set_weights, coeff_a, self.psi_a_dU_df) 

        grad_sum_j_coeff_a = np.einsum("w,wnmp,m->np", self.set_weights, self.psi_a_dU_df, coeff_U) 
        grad_sum_j_coeff_a += np.einsum("w,wnp->np", self.set_weights, self.psi_da_df)
        grad_sum_j_coeff_a += np.einsum("w,wnp->np", self.set_weights, self.psi_a_d2f)

        if self.softplus_coeff_a:
            grad_coeff_a = np.einsum("p,np,n->n", Sum_j, grad_sum_j_coeff_a, grad_coeff_a)
        else:
            grad_coeff_a = np.einsum("p,np->n", Sum_j, grad_sum_j_coeff_a)

        grad_coeff_U = np.einsum("p,mp->m", Sum_j, grad_sum_j_coeff_U)
        grad_loss = np.concatenate([grad_coeff_U, grad_coeff_a])

        return grad_loss

    def _eval_hess_loss_general_a(self, coeff, dummy):

        coeff_U = coeff[:self.R_U]
        if self.softplus_coeff_a:
            coeff_a = np.log(1 + np.exp(np.copy(coeff[self.R_U:])))
            grad_coeff_a = np.exp(coeff[self.R_U:])/(1 + np.exp(coeff[self.R_U:]))
        else:
            coeff_a = coeff[self.R_U:]
            
        Sum_j = np.einsum("w,n,wnmp,m->p", self.set_weights, coeff_a, self.psi_a_dU_df, coeff_U) 
        Sum_j += np.einsum("w,n,wnp->p", self.set_weights, coeff_a, self.psi_da_df)
        Sum_j += np.einsum("w,n,wnp->p", self.set_weights, coeff_a, self.psi_a_d2f)
        Sum_j += np.einsum("w,wp->p", self.set_weights, self.psi_f)

        n_params = self.R_U + self.R_a
        hess_loss = np.zeros((n_params, n_params), float)

        hess_loss[:self.R_U, self.R_U:] = np.einsum("w,p,wnmp->nm", self.set_weights, Sum_j, self.psi_a_dU_df)

        grad_sum_j_coeff_U = np.einsum("w,n,wnmp->mp", self.set_weights, coeff_a, self.psi_a_dU_df) 

        grad_sum_j_coeff_a = np.einsum("w,wnmp,m->np", self.set_weights, self.psi_a_dU_df, coeff_U) 
        grad_sum_j_coeff_a += np.einsum("w,wnp->np", self.set_weights, self.psi_da_df)
        grad_sum_j_coeff_a += np.einsum("w,wnp->np", self.set_weights, self.psi_a_d2f)

        grad_product = np.einsum("mp,np->nm", grad_sum_j_coeff_U, grad_sum_j_coeff_a)
        hess_loss[:self.R_U, self.R_U:] += grad_product 
        hess_loss[self.R_U:, :self.R_U] += grad_product.T
        hess_loss[self.R_U:, self.R_U:] += np.einsum("mp,np->nm", grad_sum_j_coeff_U, grad_sum_j_coeff_U)
        hess_loss[:self.R_U, :self.R_U] += np.einsum("mp,np->nm", grad_sum_j_coeff_a, grad_sum_j_coeff_a)

        if self.softplus_coeff_a:
            hess_loss[self.R_U:, :] = np.einsum("n,nm->nm", grad_coeff_a, hess_loss[self.R_U:, :])
            hess_loss[:, self.R_U:] = np.einsum("n,mn->mn", grad_coeff_a, hess_loss[:, self.R_U:])

        return hess_loss

class LinearSpectralLoss(CrossValidatedLoss):

    def __init__(self, topfile, trajnames, savedir, **kwargs):
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
        CrossValidatedLoss.__init__(self, topfile, trajnames, savedir, **kwargs)

        self.suffix = "EG"
        self.matrices_estimated = False

        if self.matrix_files_exist() and not self.recalc:
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

            if self.save_by_traj:
                tname = self.trajnames[n]
                idx1 = (os.path.dirname(tname)).split("_")[-1]
                idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
                files_out = [ self.Xname_by_traj(idx1, idx2, k + 1) for k in range(self.n_cv_sets) ]
                files_out += [ self.dname_by_traj(idx1, idx2, k + 1) for k in range(self.n_cv_sets) ]
                files_out.append(self.frame_set_name_by_traj(idx1, idx2))
                files_out_exist = [ os.path.exists(fname) for fname in files_out ]
                if np.all(files_out_exist) and not self.recalc:
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

            if self.save_by_traj:
                # save
                for k in range(self.n_cv_sets):
                    np.save(self.Xname_by_traj(idx1, idx2, k + 1), X[k])
                    np.save(self.dname_by_traj(idx1, idx2, k + 1), d[k])
                np.save(self.frame_set_name_by_traj(idx1, idx2), self.cv_set_assignment[n])

                N_prev = np.zeros(self.n_cv_sets, float)
                d = np.zeros((self.n_cv_sets, P), float)
                if Ucg.fixed_a_coeff:
                    X = np.zeros((self.n_cv_sets, P, R), float)
                else:
                    X = np.zeros((self.n_cv_sets, P, R + 1), float)

        if not self.save_by_traj:
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

    def __init__(self, topfile, trajnames, savedir, **kwargs):
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
        CrossValidatedLoss.__init__(self, topfile, trajnames, savedir, **kwargs)

        self.suffix = "FM"
        self.matrices_estimated = False

        if self.matrix_files_exist() and not self.recalc:
            self._load_matrices()

    def calc_matrices(self, Ucg, forcenames, coll_var_names=None, verbose=True, include_trajs=[], chunksize=1000):
        """Calculate force-matching matrices 
       
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

        Description
        -----------
        Using running QR decomposition to calculate.

        """

        self.Ucg = Ucg

        if self.n_cv_sets is None:
            self.n_cv_sets = 1
        else:
            if self.n_cv_sets > 1 and not self.cv_sets_are_assigned:
                self.assign_crossval_sets()

        n_params = Ucg.n_tot_params
        P = Ucg.n_test_funcs
        max_rows = chunksize*Ucg.n_dof
        A_b_set = {}

        if Ucg.using_cv and not Ucg.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(forcenames) != len(self.trajnames):
            raise ValueError("Need forces for every trajectory!")

        if len(include_trajs) > 0:
            n_trajs = len(include_trajs)
        else:
            n_trajs = len(self.trajnames)

        count = 0
        for n in range(len(self.trajnames)):
            if n in include_trajs:
                count += 1
            else:
                continue

            if self.save_by_traj:
                tname = self.trajnames[n]
                idx1 = (os.path.dirname(tname)).split("_")[-1]
                idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
                files_out = [ self.Xname_by_traj(idx1, idx2, k + 1) for k in range(self.n_cv_sets) ]
                files_out += [ self.dname_by_traj(idx1, idx2, k + 1) for k in range(self.n_cv_sets) ]
                files_out.append(self.frame_set_name_by_traj(idx1, idx2))
                files_out_exist = [ os.path.exists(fname) for fname in files_out ]
                if np.all(files_out_exist) and not self.recalc:
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


                        A[:R.shape[0],:] = R.copy()
                        b[:R.shape[0]] = np.dot(Q.T, f_target)
                    else:
                        # augment matrix system with next chunk of data
                        A[n_params:n_params + n_rows,:] = f_cg 
                        b[n_params:n_params + n_rows] = f_target

                        Q_next, R_next = scl.qr(A, mode="economic")

                        A[:R_next.shape[0],:] = R_next
                        b[:R_next.shape[0]] = np.dot(Q_next.T, b)
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

                                rows_fill = R.shape[0]

                                A[:rows_fill,:] = R.copy()
                                b[:rows_fill] = np.dot(Q.T, f_target_subset)

                                A_b_set[str(k)] = (A, b)
                            else:
                                # augment matrix system with next chunk of data
                                (A, b) = A_b_set[str(k)]
                                A[rows_fill:rows_fill + n_rows_set,:] = f_cg_subset
                                b[rows_fill:rows_fill + n_rows_set] = f_target_subset

                                Q_next, R_next = scl.qr(A, mode="economic")

                                A[:R_next.shape[0],:] = R_next
                                b[:R_next.shape[0]] = np.dot(Q_next.T, b)

                start_idx += N_chunk

            if self.save_by_traj:
                # save matrices for this traj alone
                for k in range(self.n_cv_sets):
                    np.save(self.Xname_by_traj(idx1, idx2, k + 1), A_b_set[str(k)][0])
                    np.save(self.dname_by_traj(idx1, idx2, k + 1), A_b_set[str(k)][1])
                np.save(self.frame_set_name_by_traj(idx1, idx2), self.cv_set_assignment[n])

                A_b_set = {}

        if not self.save_by_traj:
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

class LinearProjectedForceMatchingLoss(CrossValidatedLoss):

    def __init__(self, topfile, trajnames, savedir, **kwargs):
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
        CrossValidatedLoss.__init__(self, topfile, trajnames, savedir, **kwargs)

        self.suffix = "pFM"
        self.matrices_estimated = False

        if self.matrix_files_exist() and not self.recalc:
            self._load_matrices()

    def calc_matrices(self, Ucg, forcenames, coll_var_names=None, verbose=True, include_trajs=[], chunksize=1000):
        """Calculate force-matching matrices 
       
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

        Description
        -----------
        Using running QR decomposition to calculate.

        """

        raise NotImplementedError

        self.Ucg = Ucg

        if self.n_cv_sets is None:
            self.n_cv_sets = 1
        else:
            if self.n_cv_sets > 1 and not self.cv_sets_are_assigned:
                self.assign_crossval_sets()

        n_params = Ucg.n_tot_params
        P = Ucg.n_test_funcs
        max_rows = chunksize*Ucg.n_dof
        #A_b_set = {}

        if Ucg.using_cv and not Ucg.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(forcenames) != len(self.trajnames):
            raise ValueError("Need forces for every trajectory!")

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

            if self.save_by_traj:
                tname = self.trajnames[n]
                idx1 = (os.path.dirname(tname)).split("_")[-1]
                idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
                files_out = [ self.Xname_by_traj(idx1, idx2, k + 1) for k in range(self.n_cv_sets) ]
                files_out += [ self.dname_by_traj(idx1, idx2, k + 1) for k in range(self.n_cv_sets) ]
                files_out.append(self.frame_set_name_by_traj(idx1, idx2))
                files_out_exist = [ os.path.exists(fname) for fname in files_out ]
                if np.all(files_out_exist) and not self.recalc:
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

                grad_f = test_funcs_gradient(xyz_traj, cv_traj)

                dot_gU1_gf = np.einsum("tdr,tdp->trp", U1_force, grad_f)
                dot_Fmd_gf = np.einsum("td,tdp->tp", f_target_chunk, grad_f)

                if self.n_cv_sets == 1: 

                    curr_X = np.sum(dot_gU1_gf, axis=0)

                else:
                    for k in range(self.n_cv_sets):   
                        frames_in_this_set = self.cv_set_assignment[n][start_idx:start_idx + N_chunk] == k
                        n_frames_set = np.sum(frames_in_this_set)

                        if n_frames_set > 0:
                            N_prev[k] += n_frames_set 

                            # average over these frames


                start_idx += N_chunk

            if self.save_by_traj:
                pass
            #    # save matrices for this traj alone
            #    for k in range(self.n_cv_sets):
            #        np.save(self.Xname_by_traj(idx1, idx2, k + 1), A_b_set[str(k)][0])
            #        np.save(self.dname_by_traj(idx1, idx2, k + 1), A_b_set[str(k)][1])
            #    np.save(self.frame_set_name_by_traj(idx1, idx2), self.cv_set_assignment[n])

            #    A_b_set = {}

        if not self.save_by_traj:
            pass
            #if self.n_cv_sets > 1:
            #    self.X_sets = [ A_b_set[str(k)][0] for k in range(self.n_cv_sets) ]
            #    self.d_sets = [ A_b_set[str(k)][1] for k in range(self.n_cv_sets) ]
            #    self.X = np.sum([ self.set_weights[j]*self.X_sets[j] for j in range(self.n_cv_sets) ], axis=0)
            #    self.d = np.sum([ self.set_weights[j]*self.d_sets[j] for j in range(self.n_cv_sets) ], axis=0)
            #else:
            #    self.X = A
            #    self.d = b

            #self.matrices_estimated = True
            #self._save_matrices()
            #self._training_and_validation_matrices()
