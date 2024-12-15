from __future__ import print_function, division

import numpy as np

from time import time
from itertools import cycle, product

from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.utils import gen_batches
from harmonization.elastic_net import lasso_path, lasso_crossval


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


def solve_l1(X, D, alpha=None, return_all=False, nlambdas=100, ncores=-1, positivity=False, variance=None, fit_intercept=True, standardize=True,
             progressbar=True, pool=None, use_joblib=False, use_crossval=False, verbose=5, pre_dispatch='all'):

    if alpha is None:
        return_alpha = True
        alpha = np.zeros((D.shape[1], X.shape[0]))
    else:
        return_alpha = False

    if variance is None:
        variance = [None] * alpha.shape[1]

    Xhat = np.zeros((alpha.shape[1], D.shape[0]), dtype=np.float32)
    intercept = np.zeros((alpha.shape[1], 1), dtype=np.float32)
    lbda = np.zeros((alpha.shape[1], 1), dtype=np.float32)

    if use_joblib:
        stuff = Parallel(n_jobs=ncores,
                         pre_dispatch=pre_dispatch,
                         verbose=verbose)(delayed(lasso_path_parallel)(D,
                                                                       X[i],
                                                                       nlambdas=nlambdas,
                                                                       positivity=positivity,
                                                                       variance=variance[i],
                                                                       fit_intercept=fit_intercept,
                                                                       standardize=standardize,
                                                                       use_crossval=use_crossval) for i in range(alpha.shape[1]))
    else:
        raise ValueError('Only joblib path is supported now.')

    for i, content in enumerate(stuff):
        Xhat[i], alpha[:, i], intercept[i], lbda[i] = content

    if return_all:
        return Xhat, alpha, intercept, lbda

    if return_alpha:
        return alpha


def online_DL(X, D=None, n_atoms=None, niter=250, batchsize=128, rho=1., t0=1e-3, variance=None,
              shuffle=True, fulldraw=False, positivity=False, fit_intercept=True, standardize=True, ncores=-1, nlambdas=100,
              progressbar=True, disable_mkl=True, saveback=None, use_joblib=False, method='fork',
              eps=1e-6):

    tt = time()
    seen_patches = 0

    if n_atoms is None:
        n_atoms = 2 * np.prod(X.shape[1:])

    if n_atoms > X.shape[0]:
        n_atoms = X.shape[0]

    if D is None:
        # we pick n_atoms indexes along each dimensions randomly
        indices = np.random.permutation(X.shape[0])[:n_atoms]
        D = X[indices].reshape(n_atoms, np.prod(X.shape[1:])).T

    # Put to unit l2 norm, will also copy D if we passed it to the function
    D = D / np.sqrt(np.sum(D**2, axis=1, keepdims=True))

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

    for t, batch in tqdm(iterator, total=niter):

        cutter = X[batch].shape[0]
        if cutter < batchsize:
            continue

        x[:] = X[batch].reshape(batchsize, -1)
        _, _, _, lbda = solve_l1(x, D, alpha, positivity=positivity, ncores=ncores, nlambdas=nlambdas, variance=variance, return_all=True,
                                 fit_intercept=fit_intercept, standardize=standardize, use_joblib=use_joblib, method=method,
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

        # reset past information every two full epochs
        if seen_patches > 2 * X.shape[0]:
            A[:] = A_prime
            B[:] = B_prime

            A_prime[:] = 0
            B_prime[:] = 0

            seen_patches = 0
        else:
            seen_patches += batchsize

    print('total {}'.format(time() - tt))
    return D
