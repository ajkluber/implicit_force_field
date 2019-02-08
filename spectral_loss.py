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


class LinearLoss(object):

    def __init__(self):
        """Creates matrices for minimizing the linear spectral loss equations"""

        self.cv_method = None
        self.cv_sets_are_assigned = False
        self.is_estimated = False

    def assign_crossval_sets(self, trajnames, n_cv_sets=5, method="shuffled"):
        """Assign trajectories or frames in training and validation sets

        Parameters
        ----------
        trajnames : list, str
            List of trajectory filenames.

        n_cv_sets : int, default=5
            Desired number of trajectory sets.

        method : str
            shuffled = Randomly assign trajectory frames to sets. (default)
            continous = Assign continous trajs to sets.
        """

        if not method in ["shuffled", "continuous"]:
            raise ValueError("method must be be 'shuffled' or 'continuos'. Entered: " + method)

        if method == "shuffled":
            set_assignment = []
            traj_n_frames = []
            for n in range(len(trajnames)):
                length = 0
                for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                    length += chunk.n_frames
                traj_n_frames.append(length)

                set_assignment.append(np.random.randint(low=0, high=n_cv_sets, size=length))
            total_n_frames = sum(traj_n_frames)

            self.cv_method = method
            self.cv_frame_assignment = set_assignment
            self.cv_sets_are_assigned = True
        elif method == "continuous":
            raise NotImplementedError

            traj_n_frames = []
            for n in range(len(trajnames)):
                length = 0
                for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                    length += chunk.n_frames
                traj_n_frames.append(length)
            total_n_frames = sum(traj_n_frames)

            n_frames_in_set = total_n_frames/n_cv_sets

            traj_set = []
            psi_set = []
            traj_set_frames = []
            temp_traj_names = []
            temp_psi_names = []
            temp_frame_count = 0
            for n in range(len(traj_n_frames)):
                if temp_frame_count >= n_frames_in_set:
                    # finish a set when it has desired number of frames
                    traj_set.append(temp_traj_names)
                    psi_set.append(temp_psi_names)
                    traj_set_frames.append(temp_frame_count)

                    # start over
                    temp_traj_names = [trajnames[n]]
                    #temp_psi_names = [psinames[n]]
                    temp_frame_count = traj_n_frames[n]
                else:
                    temp_traj_names.append(trajnames[n])
                    #temp_psi_names.append(psinames[n])
                    temp_frame_count += traj_n_frames[n]

                if n == len(traj_n_frames) - 1:
                    traj_set.append(temp_traj_names)
                    psi_set.append(temp_psi_names)
                    traj_set_frames.append(temp_frame_count)

            with open("traj_sets_{}.txt".format(n_cv_sets), "w") as fout:
                for i in range(len(traj_set)):
                    info_str = str(traj_set_frames[i])
                    info_str += " " + " ".join(traj_set[i]) + "\n"
                    fout.write(info_str)

            #return traj_set, traj_set_frames, psi_set

    def calc_matrices(self, trajnames, topfile, psinames, ti_file, M=M, cv_names=psinames, set_assignment=set_assignment, verbose=True, a_coeff=None):
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

        if not a_coeff is None:
            self.set_fixed_diffusion_coefficient(a_coeff)

        if not set_assignment is None:
            n_sets = np.max([ np.max(x) for x in set_assignment]) + 1
        else:
            n_sets = 1

        R = self.n_params
        P = self.n_test_funcs

        # if constant diff coeff
        if self.constant_diff:
            d = np.zeros((n_sets, P), float)
            if self.fixed_diff:
                X = np.zeros((n_sets, P, R), float)
                D2 = np.zeros((n_sets, R, R), float)    # TODO: high-dimensional smoothness 
            else:
                X = np.zeros((n_sets, P, R + 1), float)
                D2 = np.zeros((n_sets, R + 1, R + 1), float)    # TODO: high-dimensional smoothness 
        else:
            raise NotImplementedError("Only constant diffusion coefficient is supported.")

        if self.using_cv and not self.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        kappa = 1./np.load(ti_file)[:M]

        N_prev = 0
        N_prev_set = np.zeros(n_sets, int)
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

                # calculate test function values, gradient, and Laplacian
                test_f = self.test_functions(xyz_traj, cv_chunk)
                grad_f, Lap_f = self.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

                if self.fixed_diff:
                    grad_U1 *= self.a_coeff
                    Lap_f *= self.a_coeff

                if self.using_U0:
                    grad_U0 = self.gradient_U0(xyz_traj, cv_chunk)
                    if self.fixed_diff:
                        grad_U0 *= self.a_coeff

                if self.using_D2:
                    #TODO
                    pass

                if n_sets == 1: 
                    # dot products with eigenvectors
                    curr_X1 = np.einsum("tm,tdr,tdp->mpr", psi_chunk, -grad_U1, grad_f).reshape((M*P, R))
                    curr_X2 = np.einsum("m,tm,tp->mp", kappa, psi_chunk, test_f).reshape(M*P)

                    curr_d = (-1./self.beta)*np.einsum("tm,tp->mp", psi_chunk, Lap_f).reshape(M*P)

                    if self.using_U0:
                        curr_d += np.einsum("tm,td,tdp->mp", psi_chunk, grad_U0, grad_f).reshape(M*P)

                    if self.fixed_diff:
                        # running average to reduce numerical error
                        curr_d -= curr_X2

                        X[0,:] = (curr_X1 + float(N_prev)*X[0,:,:-1])/(float(N_prev) + float(N_chunk))
                        d[0,:] = (curr_d + float(N_prev)*d[0,:])/(float(N_prev) + float(N_chunk))
                    else:
                        # running average to reduce numerical error
                        X[0,:,:-1] = (curr_X1 + float(N_prev)*X[0,:,:-1])/(float(N_prev) + float(N_chunk))
                        X[0,:,-1] = (curr_X2 + float(N_prev)*X[0,:,-1])/(float(N_prev) + float(N_chunk))
                        d[0,:] = (curr_d + float(N_prev)*d[0,:])/(float(N_prev) + float(N_chunk))

                    N_prev += N_chunk
                else:
                    for k in range(n_sets):   
                        frames_in_this_set = set_assignment[n][start_idx:start_idx + N_chunk] == k

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

                            curr_d = (-1./self.beta)*np.einsum("tm,tp->mp", psi_subset, Lap_f_subset).reshape(M*P)

                            if self.using_U0:
                                gU0_subset = grad_U0[frames_in_this_set]
                                curr_d += np.einsum("tm,td,tdp->mp", psi_subset, gU0_subset, gradf_subset).reshape(M*P)

                            if self.fixed_diff:
                                # running average to reduce numerical error
                                curr_d -= curr_X2

                                X[k,:] = (curr_X1 + float(N_prev_set[k])*X[k,:])/(float(N_prev_set[k] + N_curr_set))
                                d[k,:] = (curr_d + float(N_prev_set[k])*d[k,:])/(float(N_prev_set[k] + N_curr_set))
                            else:
                                # running average to reduce numerical error
                                X[k,:,:-1] = (curr_X1 + float(N_prev_set[k])*X[k,:,:-1])/(float(N_prev_set[k] + N_curr_set))
                                X[k,:,-1] = (curr_X2 + float(N_prev_set[k])*X[k,:,-1])/(float(N_prev_set[k] + N_curr_set))
                                d[k,:] = (curr_d + float(N_prev_set[k])*d[k,:])/(float(N_prev_set[k] + N_curr_set))

                            N_prev_set[k] += N_curr_set
                start_idx += N_chunk

        self.cross_val_sets = n_sets
        self.cross_val_set_n_frames = N_prev_set
        self.eigenpair_X = X
        self.eigenpair_d = d
        self.eigenpair_D2 = D2

        self.X = np.sum([ (n_frames_in_set[j]/float(total_n_frames))*X_sets[j] for j in range(n_cross_val_sets) ], axis=0)
        self.d = np.sum([ (n_frames_in_set[j]/float(total_n_frames))*d_sets[j] for j in range(n_cross_val_sets) ], axis=0)

        self.X_sets = [ Ucg.eigenpair_X[k] for k in range(n_cross_val_sets) ]
        self.d_sets = [ Ucg.eigenpair_d[k] for k in range(n_cross_val_sets) ]
        #for k in range(n_cross_val_sets):
        #    np.save("{}/X_{}.npy".format(cg_savedir, k + 1), Ucg.eigenpair_X[k])
        #    np.save("{}/d_{}.npy".format(cg_savedir, k + 1), Ucg.eigenpair_d[k])
        #    np.save("{}/frame_set_{}.npy".format(cg_savedir,  k + 1), set_assignment[k])


class NonlinearLoss(object):
    def __init__(self):
        pass

    def nonlinear_spectral_loss(self, coeff, alpha, trajnames, topfile, psinames, ti_file, M=1, cv_names=[], verbose=False, set_assignment=None):
        """Calculate eigenpair matrices
       
        Parameters
        ----------
        coeff: np.ndarray
            Trajectory filenames

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

        raise NotImplementedError

        if not set_assignment is None:
            n_sets = np.max([ np.max(x) for x in set_assignment]) + 1
        else:
            n_sets = 1

        R = self.n_params
        P = self.n_test_funcs

        n_U_params = self.n_params 

        U_coeff = coeff[:n_U_params]
        a_coeff = coeff[n_U_params:]

        training_loss = 0
        validation_loss = 0

        if self.using_cv and not self.cv_defined:
            raise ValueError("Collective variables are not defined!")

        if len(psinames) != len(trajnames):
            raise ValueError("Need eigenvector for every trajectory!")

        kappa = 1./np.load(ti_file)[:M]

        N_prev = 0
        N_prev_set = np.zeros(n_sets, int)
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


                # calculate test function values, gradient, and Laplacian
                test_f = self.test_functions(xyz_traj, cv_chunk)
                grad_f, Lap_f = self.test_funcs_gradient_and_laplacian(xyz_traj, cv_chunk)

                Jac = self._cv_cartesian_Jacobian(xyz_traj) # tmd

                #
                noise_a_k = self.calculate_noise_basis(xyz_traj, cv_chunk)
                grad_a_k = self.gradient_noise(xyz_traj, cv_chunk)

                noise_a = np.einsum("k,tk->t", a_coeff, noise_a_k)
                grad_a = np.einsum("k,tk->t", a_coeff, grad_a_k)

                # calculate b vector
                Force = np.einsum("r,tdr->td", U_coeff, grad_U1)

                #drift_1 = np.einsum("t,td->t", noise_a, Force)
                #drift_2 = np.einsum("", (1./self.beta)*grad_a)


                if self.using_U0:
                    grad_U0 = self.gradient_U0(xyz_traj, cv_chunk)
                    Force += grad_U0

                if self.using_D2:
                    #TODO
                    pass

                if n_sets == 1: 
                    # dot products with eigenvectors
                    curr_X1 = np.einsum("tm,tdr,tdp->mpr", psi_chunk, -grad_U1, grad_f).reshape((M*P, R))
                    curr_X2 = np.einsum("m,tm,tp->mp", kappa, psi_chunk, test_f).reshape(M*P)

                    curr_d = (-1./self.beta)*np.einsum("tm,tp->mp", psi_chunk, Lap_f).reshape(M*P)

                    if self.using_U0:
                        curr_d += np.einsum("tm,td,tdp->mp", psi_chunk, grad_U0, grad_f).reshape(M*P)

                    # running average to reduce numerical error
                    X[0,:,:-1] = (curr_X1 + float(N_prev)*X[0,:,:-1])/(float(N_prev) + float(N_chunk))
                    X[0,:,-1] = (curr_X2 + float(N_prev)*X[0,:,-1])/(float(N_prev) + float(N_chunk))
                    d[0,:] = (curr_d + float(N_prev)*d[0,:])/(float(N_prev) + float(N_chunk))

                    N_prev += N_chunk
                else:
                    for k in range(n_sets):   
                        frames_in_this_set = set_assignment[n][start_idx:start_idx + N_chunk] == k

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

                            curr_d = (-1./self.beta)*np.einsum("tm,tp->mp", psi_subset, Lap_f_subset).reshape(M*P)

                            if self.using_U0:
                                gU0_subset = grad_U0[frames_in_this_set]
                                curr_d += np.einsum("tm,td,tdp->mp", psi_subset, gU0_subset, gradf_subset).reshape(M*P)

                            # running average to reduce numerical error
                            X[k,:,:-1] = (curr_X1 + float(N_prev_set[k])*X[k,:,:-1])/(float(N_prev_set[k] + N_curr_set))
                            X[k,:,-1] = (curr_X2 + float(N_prev_set[k])*X[k,:,-1])/(float(N_prev_set[k] + N_curr_set))
                            d[k,:] = (curr_d + float(N_prev_set[k])*d[k,:])/(float(N_prev_set[k] + N_curr_set))

                            N_prev_set[k] += N_curr_set
                start_idx += N_chunk

        self.cross_val_sets = n_sets
        self.cross_val_set_n_frames = N_prev_set
        self.eigenpair_X = X
        self.eigenpair_d = d
        self.eigenpair_D2 = D2
