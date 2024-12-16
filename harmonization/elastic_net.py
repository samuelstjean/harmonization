import numpy as np
import warnings

from sklearn.model_selection import KFold
from harmonization._glmnet import elnet


def lasso_path(X, y, nlam=100, fit_intercept=False, pos=False, standardize=False, weights=None, penalty=None, criterion=None):
    """return full path for Lasso"""
    return elastic_net_path(X, y, rho=1.0, nlam=nlam, weights=weights, penalty=penalty, criterion=criterion,
                            fit_intercept=fit_intercept, pos=pos, standardize=standardize)


def lasso_crossval(X, y, nlam=100, fit_intercept=False, pos=False, standardize=False,
                   weights=None, penalty=None, n_splits=3, use_min_se=True):
    """return k-fold cross validation for Lasso"""

    # In this call we find a list of lambda values
    # we then crossval using those same values on different splits down later.
    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elastic_net(X,
                                                            y,
                                                            1.,
                                                            nlam=nlam,
                                                            fit_intercept=fit_intercept,
                                                            pos=pos,
                                                            weights=weights,
                                                            vp=penalty,
                                                            standardize=standardize)
    alm = alm[:lmu]
    cv_error = np.zeros((n_splits, lmu))

    for idx, (train_idx, test_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(X)):
        beta, a0, _, _ = elastic_net_path(X[train_idx],
                                          y[train_idx],
                                          rho=1.0,
                                          ulam=alm,
                                          nlam=lmu,
                                          penalty=penalty,
                                          fit_intercept=fit_intercept,
                                          pos=pos,
                                          standardize=standardize)

        y_pred = np.dot(X[test_idx], beta) + a0
        error = np.mean((y[test_idx, None] - y_pred)**2, axis=0)
        # seems like the lambda sequence can sometimes be truncated somehow?
        cv_error[idx, :len(error)] = error

    mean_cv = np.mean(cv_error, axis=0)
    best_idx = np.argmin(mean_cv)

    use_min_se = False
    if use_min_se:
        std_cv = np.std(cv_error, axis=0, ddof=1)
        semin = mean_cv[best_idx] + std_cv[best_idx]

        # we want the smallest cv + std_cv as the new model instead of just the smallest one
        # that's the length of the vector where everything is below the upper bound on cv
        best_idx = np.sum(mean_cv <= semin)

    best_lambda = alm[best_idx]

    # redo a full fit with the best lambda value we found
    beta_best, a0_best, Xhat_best, alm_best = elastic_net_path(X,
                                                               y,
                                                               rho=1.0,
                                                               ulam=best_lambda,
                                                               nlam=1,
                                                               penalty=penalty,
                                                               fit_intercept=fit_intercept,
                                                               pos=pos,
                                                               standardize=standardize)

    return Xhat_best.squeeze(), beta_best.squeeze(), a0_best, alm_best


def elastic_net_path(X, y, rho, nlam=100, ulam=None, criterion=None, variance=None,
                     fit_intercept=False, pos=False, standardize=False, weights=None, penalty=None):
    """return full path for ElasticNet"""

    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elastic_net(X,
                                                            y,
                                                            rho,
                                                            nlam=nlam,
                                                            fit_intercept=fit_intercept,
                                                            pos=pos,
                                                            ulam=ulam,
                                                            weights=weights,
                                                            vp=penalty,
                                                            standardize=standardize)
    nobs, nx = X.shape

    if jerr > 0:  # jerr < 0 is a non fatal error
        warnings.warn('Non zero error code {}'.format(jerr))

    a0 = a0[:lmu]
    ca = ca[:nx, :lmu]
    ia = ia[:nx]
    nin = nin[:lmu]
    rsq = rsq[:lmu]
    alm = alm[:lmu]

    if len(nin) == 0:
        ninmax = None
    else:
        ninmax = max(nin)

    if ninmax is None:
        return (np.zeros([nx, nlam], dtype=np.float32),
                np.zeros((nlam), dtype=np.float32),
                np.zeros([nobs, nlam], dtype=np.float32),
                np.zeros((nlam), dtype=np.float32))

    ca = ca[:ninmax]
    ja = ia[:ninmax] - 1    # ia is 1-indexed in fortran
    oja = np.argsort(ja)
    ja1 = ja[oja]
    beta = np.zeros([nx, lmu], dtype=np.float32)
    beta[ja1] = ca[oja]

    yhat = np.dot(X, beta) + a0

    # no criterion - just return the whole path
    # else we choose the best value of lambda and just return that
    if criterion is None:
        return beta, a0, yhat, alm

    n = y.shape[0]
    p = X.shape[1]

    if criterion == 'aic' or criterion == 'aicc':
        w = 2
    elif criterion == 'bic':
        w = np.log(n)
    elif criterion == 'ric':
        w = 2 * np.log(p)
    else:
        raise ValueError('Criterion {} is not supported!'.format(criterion))

    mse = np.mean((y[:, None] - yhat)**2, axis=0, dtype=np.float32)
    df_mu = np.sum(beta != 0, axis=0, dtype=np.float32)

    # criterion from burnham and anderson
    criterion_value = np.array(n * np.log(mse) + w * df_mu, dtype=np.float64)

    # criterion according to 2.15 and 2.16 of https://projecteuclid.org/download/pdfview_1/euclid.aos/1194461726
    # if variance is None:
    #     variance = np.var(y)
    # criterion_value = mse / variance + w * df_mu / n

    if criterion == 'aicc':
        with np.errstate(divide='ignore'):
            aicc = (2 * df_mu * (df_mu + 1)) / (n - df_mu - 1)
        criterion_value += aicc

    criterion_value[df_mu == 0] = 1e300
    criterion_value[np.isnan(criterion_value)] = 1e300
    best_idx = np.argmin(criterion_value, axis=0)

    # Seems like the memory is not correctly freed internally when we call from fortran,
    # so we copy everything once to not keep references to it during the parallel processing
    # or whatever, computer sciency stuff, it works :/
    beta = beta[:, best_idx].copy()
    yhat = yhat[:, best_idx].copy()
    a0 = a0[best_idx].copy()
    alm = alm[best_idx].copy()

    return beta, a0, yhat, alm


def elastic_net(X, y, rho, pos=False, thr=1e-7, weights=None, vp=None, copy=True, ulam=None, jd=np.zeros(1),
                standardize=False, nlam=100, maxit=100000, flmin=1e-4, fit_intercept=False, custom_path=False):
    """
    Raw-output wrapper for elastic net linear regression.
    X is D
    y is X
    rho for lasso/elastic net tradeoff
    """

    # X/y is overwritten in the fortran function at every loop, so we must copy it each time
    if copy:
        X = np.array(X, copy=True, dtype=np.float64, order='F')
        y = np.array(y, copy=True, dtype=np.float64, order='F').ravel()

    box_constraints = np.zeros((2, X.shape[1]), order='F')
    box_constraints[1] = 9.9e35

    if not pos:
        box_constraints[0] = -9.9e35

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(X.shape[0], order='F')
    else:
        weights = weights.copy(order='F')

    # Uniform penalties if none were specified.
    if vp is None:
        vp = np.ones(X.shape[1], order='F')
    else:
        vp = vp.copy(order='F')

    if (X.shape[1] > 500) or (X.shape[1] >= 2*X.shape[0]):
        # use naive algorithm
        ka = 2
    else:
        # use covariance algorithm
        ka = 1

    if custom_path:
        ulam = np.logspace(2, -5, num=nlam)

    # User activated values are used only if flmin >= 1.
    if ulam is not None:
        flmin = 2.

    nvars = X.shape[1]
    ne = nvars + 1
    nx = int(min(ne * 2 + 20, X.shape[1]))

    if y.ndim == 1:
        lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elnet(rho,
                                                          X,
                                                          y,
                                                          weights,
                                                          jd,
                                                          vp,
                                                          box_constraints,
                                                          nx,
                                                          flmin,
                                                          ulam,
                                                          thr,
                                                          nlam=nlam,
                                                          ka=ka,
                                                          # ne=ne,
                                                          isd=standardize,
                                                          maxit=maxit,
                                                          intr=fit_intercept)
    else:
        raise ValueError('y needs to be 1D but is {}'.format(y.ndim))

    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr
