from __future__ import print_function, absolute_import
import os
import glob
import argparse
import numpy as np
import scipy.linalg as scl
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field as iff
import implicit_force_field.polymer_scripts.util as util
import implicit_force_field.spectral_loss as spl

from implicit_force_field.spectral_loss import CrossValidatedLoss

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
        CrossValidatedLoss.__init__(self, savedir, n_cv_sets=n_cv_sets)

        self.matrices_estimated = False
        self.recalc = recalc

        if self.matrix_files_exist() and not recalc:
            self._load_matrices()

    def calc_matrices(self, Ucg, forcenames, coll_var_names=None, verbose=True):
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

        if len(forcenames) != len(self.trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        A_set = {}
        b_set = {}

        chunksize = 1000
        max_rows = chunksize*Ucg.n_dof

        N_prev = np.zeros(self.n_cv_sets, float)
        for n in range(len(self.trajnames)):
            if verbose:
                if n == len(self.trajnames) - 1:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d} DONE".format(n + 1, len(self.trajnames)))
                else:
                    print("eigenpair matrix from traj: {:>5d}/{:<5d}".format(n + 1, len(self.trajnames)), end="\r")
                sys.stdout.flush()

            # load force from simulation 
            force_traj = np.load(forcenames[n])

            if len(coll_var_names) > 0:
                # load collective variable if given
                cv_traj = np.array([ np.load(temp_cvname) for temp_cvname in coll_var_names[n] ]).T
            else:
                cv_traj = None

            # calculate matrix for trajectory
            start_idx = 0
            for chunk in md.iterload(self.trajnames[n], top=topfile, chunk=chunksize):
                N_chunk = chunk.n_frames
                n_rows = N_chunk*Ucg.n_dof

                f_target_chunk = force_traj[start_idx:start_idx + N_chunk,:]

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
                    f_cg = np.reshape(U1_force, (n_rows, R))
                    f_target = np.reshape(f_target_chunk, (n_rows, R))

                    if iteration_idx == 0:
                        Q, R = scl.qr(f_cg, mode="economic")

                        A = np.zeros((R + max_rows, R), float)
                        b = np.zeros(R + max_rows, float)

                        A[:R,:] = R[:R,:].copy()
                        b[:R] = np.dot(Q.T, f_target)
                    else:
                        # augment matrix system with next chunk of data
                        A[R:R + n_rows,:] = f_cg 
                        b[R:R + n_rows] = f_target

                        Q_next, R_next = scl.qr(A, mode="economic")

                        A[:R,:] = R_next
                        b[:R] = np.dot(Q_next.T, b)
                    N_prev[0] += float(N_chunk)
                else:
                    for k in range(self.n_cv_sets):   
                        frames_in_this_set = self.cv_set_assignment[n][start_idx:start_idx + N_chunk] == k
                        n_frames_set = np.sum(frames_in_this_set)
                        n_rows_set = n_frames_set*Ucg.n_dof

                        if n_frames_set > 0:
                            f_cg_subset = np.reshape(U1_force[frames_in_this_set], (n_frames_set*Ucg.n_dof, R))  
                            f_target_subset = np.reshape(f_target_chunk[frames_in_this_set], (n_frames_set*Ucg.n_dof, R))  

                            if not Q_set.has_key(str(k)):
                                Q, R = scl.qr(f_cg, mode="economic")

                                A = np.zeros((R + max_rows, R), float)
                                b = np.zeros(R + max_rows, float)

                                A[:R,:] = R[:R,:].copy()
                                b[:R] = np.dot(Q.T, f_target)

                                A_set[str(k)] = (A, b)
                            else:
                                # augment matrix system with next chunk of data
                                (A, b) = A_set[str(k)]
                                A[R:R + n_rows_set,:] = f_cg 
                                b[R:R + n_rows_set] = f_target

                                Q_next, R_next = scl.qr(A, mode="economic")

                                A[:R,:] = R_next
                                b[:R] = np.dot(Q_next.T, b)

                            N_prev[k] += float(n_frames_set)
                start_idx += N_chunk

        if self.n_cv_sets > 1:
            self.X_sets = [ A_set[str(k)] for k in range(self.n_cv_sets) ]
            self.d_sets = [ b_set[str(k)] for k in range(self.n_cv_sets) ]
            self.X = np.sum([ self.set_weights[j]*self.X_sets[j] for j in range(self.n_cv_sets) ], axis=0)
            self.d = np.sum([ self.set_weights[j]*self.d_sets[j] for j in range(self.n_cv_sets) ], axis=0)
        else:
            self.X = A
            self.d = b

        self.matrices_estimated = True
        self._save_matrices()
        self._training_and_validation_matrices()

    def matrix_files_exist(self):
        X_files_exist = np.all([ os.path.exists("{}/X_FM_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ])
        d_files_exist = np.all([ os.path.exists("{}/d_FM_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ])
        set_files_exist = np.all([ os.path.exists("{}/frame_set_FM_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ])
        files_exist = X_files_exist and d_files_exist and set_files_exist

        return files_exist

    def _save_matrices(self): 
        for k in range(self.n_cv_sets):
            np.save("{}/X_FM_{}.npy".format(self.savedir, k + 1), self.X_sets[k])
            np.save("{}/d_FM_{}.npy".format(self.savedir, k + 1), self.d_sets[k])
            np.save("{}/frame_set_FM_{}.npy".format(self.savedir,  k + 1), self.cv_set_assignment[k])    

        np.save("{}/X_FM.npy".format(self.savedir), self.X)
        np.save("{}/d_FM.npy".format(self.savedir), self.d)

    def _load_matrices(self):
        
        if self.n_cv_sets is None:
            raise ValueErro("Need to define number of cross val sets in order to load them")

        print("Loaded saved matrices...")
        self.X_sets = [ np.load("{}/X_FM_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ]
        self.d_sets = [ np.load("{}/d_FM_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ]
        set_assignment = [ np.load("{}/frame_set_FM_{}.npy".format(self.savedir, i + 1)) for i in range(self.n_cv_sets) ]

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
        
        self.alpha_star_idx = np.argmin(valid_mse)
        self.coeff_star = coeffs[self.alpha_star_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--n_basis", type=int, default=40)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--fixed_bonds", action="store_true")
    parser.add_argument("--recalc_matrices", action="store_true")
    args = parser.parse_args()

    msm_savedir = args.msm_savedir
    M = args.psi_dims
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    fixed_bonded_terms = args.fixed_bonds
    recalc_matrices = args.recalc_matrices

    #python ~/code/implicit_force_field/force_matching_soln.py msm_dists --psi_dims 1 --n_basis 40 --n_test 100 --fixed_bonds

    n_beads = 25 

    forcefile = "c25_forces_1.dat"

    n_beads = 25
    n_dim = 3*n_beads
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    n_pair_gauss = 10

    using_cv = True
    using_cv_r0 = False

    using_D2 = False
    n_cross_val_sets = 5

    #print("building basis function database...")
    Ucg, cg_savedir, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fixed_bonded_terms, using_cv,
            using_cv_r0, using_D2, n_cv_basis_funcs, n_cv_test_funcs, 
            cg_savedir="Ucg_FM")

    # only get trajectories that have saved forces
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    forcenames = glob.glob("run_*/" + name + "_forces_*.dat") 
    rundirs = []
    psinames = []
    trajnames = []
    for i in range(len(forcenames)):
        fname = forcenames[i]
        idx1 = (os.path.dirname(fname)).split("_")[-1]
        idx2 = (os.path.basename(fname)).split(".dat")[0].split("_")[-1]

        traj_name = "run_{}/{}_traj_cent_{}.dcd".format(idx1, name, idx2)
        #if not os.path.exists(traj_name):
        #    raise ValueError("Trajectory does not exist: " + traj_name)

        trajnames.append(traj_name)

        temp_names = []
        for n in range(M):
            temp_names.append(msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, n+1))
        psinames.append(temp_names)

    cg_savedir = cg_savedir + "_crossval_{}".format(n_cross_val_sets)

    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)
    print(cg_savedir)

    raise SystemExit

    ##################################################################
    # calculate matrix X and d 
    ##################################################################
    s_loss = spl.LinearForceMatchingLoss(topfile, trajnames, cg_savedir, n_cv_sets=n_cross_val_sets, recalc=recalc_matrices)

    if not s_loss.matrix_files_exist() or recalc_matrices:
        s_loss.assign_crossval_sets(topfile, trajnames, n_cv_sets=n_cross_val_sets, method="shuffled")
        s_loss.calc_matrices(Ucg, topfile, trajnames, psinames, ti_file, M=M, coll_var_names=psinames, verbose=True)

