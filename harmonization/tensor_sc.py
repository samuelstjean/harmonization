import numpy as np
# import mkl

from time import time
from itertools import cycle, product
from sklearn.utils import gen_batches
from sklearn.linear_model import MultiTaskLassoCV
# from sklearn.utils import shuffle as shuffler

from joblib import Parallel, delayed
from multiprocessing import get_context, cpu_count

from tqdm import tqdm

from enet import lasso_path, lasso_crossval  # select_best_path,

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/coordinate_descent.py


# # See p. 4 of https://web.stanford.edu/~hastie/Papers/multi_response.pdf
def multilasso(X, y, lbda, beta_init=0, eps=1e-5, maxiter=1000, intercept=False, progressbar=True, random=True):

    if lbda <= 0:
        raise ValueError('lambda needs to be strictly positive, but is {}'.format(lbda))

    if intercept:
        X = X - np.mean(X, axis=1, keepdims=True)
        y = y - np.mean(y, axis=1, keepdims=True)

    beta = np.zeros((X.shape[0], y.shape[1]))
    beta_prev = np.zeros_like(beta)

    R = y - np.dot(X, beta_init)
    Rk = np.zeros_like(R)

    if progressbar:
        maxiter = tqdm(maxiter)

    for _ in range(maxiter):

        ranger = range(X.shape[1])

        if random:
            ranger = list(ranger)
            np.random.shuffle(ranger)

        for k in ranger:

            # step 2a
            Rk[:] = R + np.sum(X[:, k] * beta[k])

            # step 2b
            XtRk = np.dot(X[:, k].T, Rk)
            norm2_XtRk = np.linalg.norm2(XtRk)

            if norm2_XtRk < lbda:
                beta[k] = 0
            else:
                beta[k] = 1 / np.sum(X[:, k]**2) * (1 - lbda / norm2_XtRk) * XtRk

            # step 2c
                R[:] = Rk - np.dot(X[:, k], beta[k])

        if np.abs(beta - beta_prev).max() < eps:
            break

        beta_prev[:] = beta

    return beta


def update_D(D, A, B, X, niter=1, eps=1e-15, positivity=False):

    u = np.zeros(D.shape)
    norm2 = np.zeros(D.shape[1])
    to_purge = np.zeros(D.shape[1], dtype=np.bool)

    for _ in range(niter):
        # divide by zeros are replaced by 0, but still raise the warning, so we squelch it
        with np.errstate(divide='ignore', invalid='ignore'):
            u[:] = np.where(np.diag(A) != 0, (B - np.dot(D, A))/np.diag(A) + D, 0)

        norm2[:] = np.sqrt(np.sum(u**2, axis=0))

        # purge unused atoms from D
        to_purge[:] = norm2 < eps

        if np.any(to_purge):
            indices = np.random.randint(X.shape[0], size=np.sum(to_purge))
            u[:, to_purge] = X[indices].reshape(np.sum(to_purge), u.shape[0]).T
            norm2[to_purge] = np.sqrt(np.sum(u[:, to_purge]**2, axis=0))

        D[:] = u / np.maximum(1., norm2)

        if positivity:
            D.clip(min=0., out=D)


def lasso_path_parallel(D, X, nlambdas, positivity=False, variance=None, fit_intercept=True, standardize=True, use_crossval=False, n_splits=3):

    if use_crossval:
        Xhat, alpha, intercept, lbda = lasso_crossval(D, X, nlam=nlambdas, fit_intercept=fit_intercept, n_splits=n_splits,
                                                      pos=positivity, standardize=standardize, penalty=None)
    else:
        alpha, intercept, Xhat, lbda = lasso_path(D, X, nlam=nlambdas, fit_intercept=fit_intercept, criterion='aic',
                                                  pos=positivity, standardize=standardize, penalty=None)

    return Xhat, alpha, intercept, lbda


def solve_l1(X, D, alpha=None, return_all=False, nlambdas=100, ncores=None, positivity=False, variance=None, fit_intercept=True, standardize=True,
             progressbar=True, pool=None, use_joblib=False, method='fork', multipredict=False, use_crossval=False):

    if alpha is None:
        return_alpha = True
        alpha = np.zeros((D.shape[1], X.shape[0]))
    else:
        return_alpha = False

    if ncores is None:
        ncores = cpu_count()

    if variance is None:
        variance = [None] * alpha.shape[1]

    # mkl.set_num_threads(1)

    multipredict = False
    if multipredict:
        # try out the multitask lasso
        reg = MultiTaskLassoCV(cv=5,
                               eps=1e-5,
                               verbose=False,
                               normalize=standardize,
                               fit_intercept=fit_intercept,
                               n_jobs=ncores,
                               selection='random')

        reg.fit(D, X.T)

        Xhat = reg.predict(D)
        alpha = reg.coefs_
        intercept = reg.intercept_
        lbda = reg.alpha_ * np.ones_like(intercept)

        if return_all:
            return Xhat, alpha, intercept, lbda

        if return_alpha:
            return alpha

    else:

        Xhat = np.zeros((alpha.shape[1], D.shape[0]), dtype=np.float32)
        intercept = np.zeros((alpha.shape[1], 1), dtype=np.float32)
        lbda = np.zeros((alpha.shape[1], 1), dtype=np.float32)

        # use_joblib = True

        if use_joblib:
            # tt = time()
            batch_size = alpha.shape[1] // (10 * ncores)
            stuff = Parallel(n_jobs=ncores,
                             pre_dispatch='all',
                             batch_size=batch_size,
                             verbose=5)(delayed(lasso_path_parallel)(D,
                                                                     X[i],
                                                                     nlambdas=nlambdas,
                                                                     positivity=positivity,
                                                                     variance=variance[i],
                                                                     fit_intercept=fit_intercept,
                                                                     standardize=standardize,
                                                                     use_crossval=use_crossval) for i in range(alpha.shape[1]))
            # print('time was {}'.format(time() - tt))
        else:
            arglist = [(D,
                        X[i],
                        nlambdas,
                        positivity,
                        variance[i],
                        fit_intercept,
                        standardize,
                        use_crossval)
                       for i in range(alpha.shape[1])]

            if progressbar:
                arglist = tqdm(arglist)
            # tt = time()
            if pool is None:
                with get_context(method=method).Pool(processes=ncores) as pool:
                    stuff = pool.starmap(lasso_path_parallel, arglist)
            else:
                stuff = pool.starmap(lasso_path_parallel, arglist)
            # print('time was {}'.format(time() - tt))
            del arglist

    for i, content in enumerate(stuff):
        Xhat[i], alpha[:, i], intercept[i], lbda[i] = content

    if return_all:
        return Xhat, alpha, intercept, lbda

    if return_alpha:
        return alpha


def online_DL(X, D=None, n_atoms=None, niter=250, batchsize=128, rho=1., t0=1e-3, variance=None,
              shuffle=True, fulldraw=False, positivity=False, fit_intercept=True, standardize=True, ncores=None, nlambdas=100,
              progressbar=True, disable_mkl=True, saveback=None, use_joblib=False, method='fork',
              eps=1e-6):

    tt = time()
    seen_patches = 0
    # obj_prev = 1e300

    if n_atoms is None:
        n_atoms = 2 * np.prod(X.shape[1:])

    if n_atoms > X.shape[0]:
        n_atoms = X.shape[0]

    if ncores is None:
        ncores = cpu_count()

    if D is None:
        # we pick n_atoms indexes along each dimensions randomly
        indices = np.random.permutation(X.shape[0])[:n_atoms]
        D = X[indices].reshape(n_atoms, np.prod(X.shape[1:])).T

    # Put to unit l2 norm, will also copy D if we passed it to the function
    D = D / np.sqrt(np.sum(D**2, axis=1, keepdims=True))
    # D_old = np.copy(D)

    A = np.eye(D.shape[1]) * t0
    B = t0 * D

    A_prime = np.copy(A)
    B_prime = np.copy(B)

    x = np.zeros((batchsize, D.shape[0]))
    alpha = np.zeros((n_atoms, batchsize))

    alpha_alpha_T = np.zeros((n_atoms, n_atoms))
    x_alpha_T = np.zeros((x.shape[1], n_atoms))

    if shuffle:
        np.random.shuffle(X)

    if fulldraw:
        # one iteration goes through all elements
        batches = gen_batches(X.shape[0], batchsize)
        iterator = product(range(1, niter + 1), batches)
    else:
        # one iteration goes through one draw of size batchsize from X
        batches = cycle(gen_batches(X.shape[0], batchsize))
        iterator = zip(range(1, niter + 1), batches)

    # create pool, we reuse it across iteration to save speed and close it at the end
    # if disable_mkl:
    #     mkl.set_num_threads(1)

    pool = get_context(method=method).Pool(processes=ncores)
    loss = np.zeros(niter, dtype=np.float32)

    for t, batch in tqdm(iterator, total=niter):

        cutter = X[batch].shape[0]
        if cutter < batchsize:
            continue

        x[:] = X[batch].reshape(batchsize, -1)
        _, _, _, lbda = solve_l1(x, D, alpha, positivity=positivity, ncores=ncores, nlambdas=nlambdas, variance=variance, return_all=True,
                                 fit_intercept=fit_intercept, standardize=standardize, pool=pool, use_joblib=use_joblib, method=method,
                                 progressbar=progressbar)

        np.dot(alpha, alpha.T, out=alpha_alpha_T)
        np.dot(x.T, alpha.T, out=x_alpha_T)  # x is transposed with regards to original notation

        beta = (1 - (1 / t))**rho

        A *= beta
        A += alpha_alpha_T / batchsize
        A_prime *= beta
        A_prime += alpha_alpha_T / batchsize

        B *= beta
        B += x_alpha_T
        B_prime *= beta
        B_prime += x_alpha_T

        update_D(D, A, B, X, positivity=False)

        # seems like we enter a loop where column are swapped, maybe check the quality of reconstruction in l2 norm
        # over a few iterations instead?
        # obj =
        # if nothing changed in D, then declare convergence
        # if np.abs(D_old - D).sum() / np.abs(D_old).sum()  < eps:
        #     print(np.abs(D_old - D).sum(), np.abs(D_old - D).sum() / np.abs(D_old).sum(), t, 'converged!')
        #     break
        # else:
        #     print(np.abs(D_old - D).sum(), np.abs(D_old - D).sum() / np.abs(D_old).sum(), t)
        #     D_old[:] = D

        # reset past information every two full epochs
        if seen_patches > 2 * X.shape[0]:
            A[:] = A_prime
            B[:] = B_prime

            A_prime[:] = 0
            B_prime[:] = 0

            seen_patches = 0
        else:
            seen_patches += batchsize

        loss[t - 1] = np.mean(0.5 * np.sum((x.T - np.dot(D, alpha))**2) + lbda * np.abs(alpha).sum())
        # loss[t-1] = np.mean(np.abs(alpha != 0).sum())

        if saveback is not None:
            np.save(saveback, D)
            np.save(saveback.replace('.npy', '_l2loss.npy'), loss)

    pool.close()
    pool.join()

    print('total {}'.format(time() - tt))
    return D
