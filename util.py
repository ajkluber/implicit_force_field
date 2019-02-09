from __future__ import print_function
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
import scipy.linalg

#from sklearn.cross_validation import KFold
import sklearn.model_selection
import sklearn.linear_model as sklin
import sklearn
print("sklearn: " + sklearn.__version__)

import mdtraj as md

def Ruiz_preconditioner(X, d):
    """Scale the columns and rows of matrix to reduce its condition number

    * Not gauranteed to lower the condition number.

    Parameters
    ----------
    X : np.array (M, N)
        Matrix to be pre-conditioned.

    d : np.array (M)
        Vector

    Returns
    -------
    d1 : np.array (M)
        Factors that scale the rows.

    d2 : np.array (N)
        Factors that scale the columns.

    pre_X : np.array (M, N)
        Pre-conditioned matrix.

    pre_d : np.array (M)
        Pre-conditioned vector.
    """

    eps1 = 1
    eps2 = 1
    pre_X = X.copy()
    d1 = np.ones(X.shape[0])
    d2 = np.ones(X.shape[1])
    row_norm = np.linalg.norm(pre_X, axis=1)
    col_norm = np.linalg.norm(pre_X, axis=0)
    r1 = np.max(row_norm)/np.min(row_norm)
    r2 = np.max(col_norm)/np.min(col_norm)

    #nm_ratio = (float(pre_X.shape[0])/float(pre_X.shape[1]))**(1/4)
    nm_ratio = (float(pre_X.shape[0])/float(pre_X.shape[1]))

    print("cond(X)    r1     r2")
    print("{:.4f}".format(np.log10(np.linalg.cond(X))))

    max_iter = 40
    iter = 1

    # Ruiz algorithm seeks to have unit norm rows and columns
    while ((r1 > eps1) and (r2 > eps2)) and (iter < max_iter):
        print("{:.4f}  {:.4f}  {:.4f}".format(np.log10(np.linalg.cond(pre_X)), r1, r2))
        d1 *= 1/np.sqrt(row_norm)
        d2 *= nm_ratio/np.sqrt(col_norm)

        # scale rows and columns
        pre_X = np.einsum("i, ij, j->ij", d1, X, d2)

        row_norm = np.linalg.norm(pre_X, axis=1)
        col_norm = np.linalg.norm(pre_X, axis=0)

        r1 = np.max(row_norm)/np.min(row_norm)
        r2 = np.max(col_norm)/np.min(col_norm)
        iter += 1
    print("{:.4f}  {:.4f}  {:.4f}".format(np.log10(np.linalg.cond(pre_X)), r1, r2))

    pre_d = d1*d

    return d1, d2, pre_X, pre_d

def D1_operator(Ucg, r, variable_noise=False):
    """First order finite differences of basis functions. For regularization"""

    # number of basis functions
    n_b = len(Ucg.b_funcs[1])
    if variable_noise:
        n_a = len(Ucg.a_funcs[1])
    else:
        n_a = 1

    D = np.zeros((len(r), n_b + n_a), float)
    for i in range(n_b + n_a):
        if i < n_b:
            y = Ucg.b_funcs[1][i](r)
        else:
            if variable_noise:
                y = Ucg.a_funcs[1][i - n_b](r)
            else:
                break
        D[:-1,i] = y[1:] - y[:-1]
        D[-1,i] = y[-1] - y[-2]

    return D/(r[1] - r[0])

def D2_operator(Ucg, r, variable_noise=False):
    """Second order finite differences of basis functions. For regularization"""

    # number of basis functions
    n_b = len(Ucg.b_funcs[1])
    if variable_noise:
        n_a = len(Ucg.a_funcs[1])
    else:
        n_a = 1

    D2 = np.zeros((len(r), n_b + n_a), float)
    for i in range(n_b + n_a):
        if i < n_b:
            y = Ucg.b_funcs[1][i](r)
        else:
            if variable_noise:
                y = Ucg.a_funcs[1][i - n_b](r)
            else:
                break

        # centered differences
        D2[1:-1,i] = (y[2:] - 2*y[1:-1] + y[:-2])

        # forward and backward difference
        D2[0,i] = (y[0] - 2*y[1] + y[2])
        D2[-1,i] = (y[-1] - 2*y[-2] + y[-3])

    return D2/((r[1] - r[0])**2)

def solve_deriv_regularized(alphas, A, b, Ucg, r, weight_a=1, order=1, variable_noise=False):
    """Solve regularized system for """
    if order == 1:
        D = D1_operator(Ucg, r, variable_noise=variable_noise)
    elif order == 2:
        D = D2_operator(Ucg, r, variable_noise=variable_noise)
    else:
        raise ValueError("order must be 1 or 2")

    n_b = len(Ucg.b_funcs[1])

    if len(Ucg.a_funcs[1]) > 0:
        D[:,n_b:] *= weight_a

    all_coeff = []
    res_norm = []
    deriv_norm = []
    for i in range(len(alphas)):
        # regularize the second derivative of solution
        A_reg = np.dot(A.T, A) + alphas[i]*np.dot(D.T, D)
        b_reg = np.dot(A.T, b)

        x = scipy.linalg.lstsq(A_reg, b_reg, cond=1e-10)[0]

        all_coeff.append(x)
        res_norm.append(np.linalg.norm(np.dot(A, x) - b))
        deriv_norm.append(np.linalg.norm(np.dot(D, x)))

    return all_coeff, res_norm, deriv_norm

def traj_chunk_cross_validated_least_squares(alphas, A, b, A_sets, b_sets, D):
    """Ridge regression with cross-validation"""

    ## DEPRECATED

    n_sets = len(A_sets)
    coeffs = []
    train_mse = []
    test_mse = []
    for i in range(len(alphas)):
        if i == len(alphas) - 1: 
            print("Solving: {:>5d}/{:<5d} DONE".format(i+1, len(alphas)))
        else:
            print("Solving: {:>5d}/{:<5d}".format(i+1, len(alphas)), end="\r")
        sys.stdout.flush()
        
        # folds are precalculated matrices on trajectory chunks
        train_mse_folds = [] 
        test_mse_folds = [] 
        for k in range(len(A_sets)):
            A_train, A_test = A_sets[k]
            b_train, b_test = b_sets[k]

            A_reg = np.dot(A_train.T, A_train) + alphas[i]*D
            b_reg = np.dot(A_train.T, b_train)
            coeff_fold = scipy.linalg.lstsq(A_reg, b_reg, cond=1e-10)[0]

            train_mse_folds.append(np.mean((np.dot(A_train, coeff_fold) - b_train)**2))
            test_mse_folds.append(np.mean((np.dot(A_test, coeff_fold) - b_test)**2))

        train_mse.append([np.mean(train_mse_folds), np.std(train_mse_folds)/np.sqrt(float(n_sets))])
        test_mse.append([np.mean(test_mse_folds), np.std(test_mse_folds)/np.sqrt(float(n_sets))])

        A_reg = np.dot(A.T, A) + alphas[i]*D
        b_reg = np.dot(A.T, b)
        coeffs.append(scipy.linalg.lstsq(A_reg, b_reg, cond=1e-10)[0])

    coeffs = np.array(coeffs)
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)
    return coeffs, train_mse, test_mse

def cross_validated_least_squares(alphas, A, b, D, n_splits=10):
    """Ridge regression with cross-validation"""

    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True) 

    coeffs = []
    train_mse = []
    test_mse = []
    for i in range(len(alphas)):
        if i == len(alphas) - 1: 
            print("Solving: {:>5d}/{:<5d} DONE".format(i+1, len(alphas)))
        else:
            print("Solving: {:>5d}/{:<5d}".format(i+1, len(alphas)), end="\r")
        sys.stdout.flush()

        train_mse_folds = [] 
        test_mse_folds = [] 
        for train_set, test_set in kf.split(A):
            A_reg = np.dot(A[train_set,:].T, A[train_set,:]) + alphas[i]*D
            b_reg = np.dot(A[train_set,:].T, b[train_set])

            coeff_fold = scipy.linalg.lstsq(A_reg, b_reg, cond=1e-10)[0]

            y_trial = np.dot(A, coeff_fold)

            train_mse_folds.append(np.mean((y_trial[train_set] - b[train_set])**2))
            test_mse_folds.append(np.mean((y_trial[test_set] - b[test_set])**2))
        train_mse.append([np.mean(train_mse_folds), np.std(train_mse_folds)/np.sqrt(float(n_splits))])
        test_mse.append([np.mean(test_mse_folds), np.std(test_mse_folds)/np.sqrt(float(n_splits))])

        coeffs.append(scipy.linalg.lstsq(A, b, cond=1e-10)[0])  # WRONG

    coeffs = np.array(coeffs)
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)
    return coeffs, train_mse, test_mse

def temp():
    I = np.identity(X.shape[1])
    rdg_coeffs = []
    for i in range(len(rdg_alphas)):
        A_reg = np.dot(X.T, X) + rdg_alphas[i]*I
        b_reg = np.dot(X.T, d)
        rdg_coeffs.append(scipy.linalg.lstsq(A_reg, b_reg, cond=1e-10)[0])

def solve_D2_regularized(alphas, A, b, D2, n_b=None, weight_a=None, variable_noise=False, n_splits=20):
    """Solve regularized system for """

    #if not weight_a is None:
    #    D2[n_b:,n_b:] *= weight_a


    # find alpha through cross-validation
    # train and test on different subsets of data
    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)

    all_coeff = []
    cv_score = []
    for i in range(len(alphas)):
        # cross-validation score is the average Mean-Squared-Error (MSE) over
        # data folds.
        cv_alpha = 0
        for train_set, test_set in kf.split(A):
            # regularize the second derivative of solution
            A_reg = np.dot(A[train_set,:].T, A[train_set,:]) + alphas[i]*D2
            b_reg = np.dot(A[train_set,:].T, b[train_set])

            coeff = np.linalg.lstsq(A_reg, b_reg, rcond=1e-11)[0]

            b_test = np.dot(A[test_set,:], coeff)

            MSE = np.mean((b_test - b[test_set])**2)
            cv_alpha += MSE

        cv_score.append(cv_alpha/float(n_folds))
        A_reg = np.dot(A.T, A) + alphas[i]*D2
        b_reg = np.dot(A.T, b)
        coeff = np.linalg.lstsq(A_reg, b_reg, rcond=1e-11)[0]

        all_coeff.append(coeff)
        res_norm.append(np.linalg.norm(np.dot(A, coeff) - b))
        deriv_norm.append(np.dot(coeff.T, np.dot(D2, coeff)))


    return np.array(all_coeff), np.array(res_norm), np.array(deriv_norm), np.array(cv_score)

def nonlinear_solver(alphas, A, b, D2, n_b):
    """Solution that ensures positive diffusion coefficient"""

    ramp = lambda c_prime: np.log(1 + np.exp(c_prime))
    d_ramp = lambda c_prime: np.exp(c_prime)/(1 + np.exp(c_prime))
    d2_ramp = lambda c_prime: ((1 + np.exp(c_prime))*np.exp(c_prime) - np.exp(2*c_prime))/((1 + np.exp(c_prime))**2)

    x0 = 0.0001*np.ones(A.shape[1])

    def nonlinear_residual(coeff):
        c = np.copy(coeff)
        c[n_b:] = ramp(c[n_b:])
        #residual = np.zeros(A.shape[0], float)
        return np.dot(A,c) - b

    def nonlinear_Jacobian(coeff):
        Jac = np.copy(A)
        Jac[:,n_b:] = np.einsum("ij,j->ij", Jac[:,n_b:], d_ramp(coeff[n_b:]))
        return Jac

    def nonlinear_model(coeff):
        c = np.copy(coeff)
        c[n_b:] = ramp(c[n_b:])
        #residual = np.zeros(A.shape[0], float)
        return np.dot(A,c)

    def objective_reg(coeff, *args):
        alpha = args[0]

        nneg_c = np.copy(coeff)
        nneg_c[n_b:] = ramp(nneg_c[n_b:])

        diff = np.dot(A,nneg_c) - d

        squared_res = np.sum(diff**2)
        reg_penalty = float(np.einsum("i,ij,j", nneg_c, D2, nneg_c))

        return squared_res + alpha*reg_penalty

    def obj_gradient_reg(coeff, *args):
        alpha = args[0]

        nneg_c = np.copy(coeff)
        nneg_c[n_b:] = ramp(nneg_c[n_b:])

        diff = np.dot(A,nneg_c) - d

        Jac = np.copy(A)
        Jac[:,n_b:] = np.einsum("ij,j->ij", Jac[:,n_b:], d_ramp(coeff[n_b:]))


        term1 = 2*np.einsum("i,ik->k", diff, Jac)
        term2 = 2*alpha*np.einsum("ik,i->k", D2, nneg_c)
        return term1 + term2


    def obj_hessian_reg(coeff, *args):
        alpha = args[0]
        N = A.shape[1]

        nneg_c = np.copy(coeff)
        nneg_c[n_b:] = ramp(nneg_c[n_b:])

        diff = np.dot(A,nneg_c) - d

        Jac = np.copy(A)
        Jac[:,n_b:] = np.einsum("ij,j->ij", Jac[:,n_b:], d_ramp(coeff[n_b:]))

        mixed_deriv = np.zeros((N, N))
        mixed_deriv[(np.arange(n_b, N), np.arange(n_b, N))] = np.einsum("i,ik,k->i", diff[n_b:], A[:,n_b:], d2_ramp(coeff[n_b:]))

        return 2*(np.einsum("il,ik", Jac, Jac) + mixed_deriv + alpha*D2)

    alphas = np.logspace(-14, -6, 10)
    all_coeff = []
    for i in range(len(alphas)):
        output = minimize(objective_reg, x0, args=(alphas[i]), tol=1e-10, options={"disp":True}, jac=obj_gradient_reg, hess=obj_hessian_reg, method="Newton-CG")
        temp_coeff = output.x
        temp_coeff[n_b:] = ramp(output.x[n_b])
        all_coeff.append(temp_coeff)

    output = minimize(objective_reg, x0, args=(alphas[i]), jac=obj_gradient_reg, hess=obj_hessian_reg, method="Newton-CG")



    #output = least_squares(nonlinear_residual, x0, jac=nonlinear_Jacobian,
    #        method="trf", tr_solver="exact", tr_options={"regularize":True}, verbose=1)

    #all_coeff = []
    #res_norm = []
    #deriv_norm = []
    #for i in range(len(alphas)):
    #    # regularize the second derivative of solution
    #    A_reg = np.dot(A.T, A) + alphas[i]*D2
    #    b_reg = np.dot(A.T, b)

    #    output = least_squares(model_residual, x0, jac=model_Jacobian)

    #    all_coeff.append(x)
    #    res_norm.append(np.linalg.norm(np.dot(A, x) - b))
    #    deriv_norm.append(np.dot(x.T, np.dot(D2, x)))

    return output

## JUNK
    #lower_bounds = np.zeros(A.shape[1])
    #lower_bounds[:n_b] = -np.inf

    #upper_bounds = np.zeros(A.shape[1])
    #upper_bounds = np.inf

    #linear_residual = lambda coeff: np.dot(A, coeff) - b
    #linear_Jacobian = lambda coeff: A
    #output = least_squares(linear_residual, x0, jac=linear_Jacobian,
    #        bounds=(lower_bounds, upper_bounds), method="trf")

def simple(alphas):

    # Ridge using Leave-one-out cross validation.
    ridge = sklin.RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True)
    ridge.fit(X,d)

    # cv score is the MSE
    cv_scores = np.mean(ridge.cv_values_, axis=0)
