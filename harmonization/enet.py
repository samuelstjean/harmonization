import numpy as np

import warnings
from _glmnet import elnet, multelnet

from sklearn.model_selection import KFold
# from sklearn.model_selection import GroupShuffleSplit

# from scipy.sparse import csc_matrix
# from scipy.special import i0
# from scipy.optimize import nnls, lsq_linear
# import _glmnet


def elastic_net_path(X, y, rho, nlam=100, ulam=None, criterion=None,
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
    # df_mu = np.sum(ca != 0, axis=0)
    ja = ia[:ninmax] - 1    # ia is 1-indexed in fortran
    oja = np.argsort(ja)
    ja1 = ja[oja]
    beta = np.zeros([nx, lmu], dtype=np.float32)
    beta[ja1] = ca[oja]

    yhat = np.dot(X, beta) + a0

    # no criterion- just return the whole path
    # else we choose the best value of lambda and just return that
    if criterion is None:
        return beta, a0, yhat, alm

    # y = Xhat
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

    squared_error = (y[..., None] - yhat)**2
    mse = np.mean(squared_error, axis=0, dtype=np.float32)
    df_mu = np.sum(beta != 0, axis=0, dtype=np.float32)

    # criterion according to 2.15 and 2.16 of https://projecteuclid.org/download/pdfview_1/euclid.aos/1194461726
    variance = None
    if variance is None:
        variance = np.var(y)
    # criterion_value = mse / variance + w/n * df_mu
    criterion_value = n * np.log(mse) + w * df_mu

    if criterion == 'aicc':
        with np.errstate(divide='ignore'):
            aicc = (2 * df_mu * (df_mu + 1)) / (n - df_mu - 1)
        criterion_value += aicc

    criterion_value[df_mu == 0] = 1e300
    criterion_value[np.isnan(criterion_value)] = 1e300
    best_idx = np.argmin(criterion_value, axis=0)

    return beta[:, best_idx], a0[best_idx], yhat[:, best_idx], alm[best_idx]


# def select_best_path(X, y, beta, mu, variance=None, criterion='aic', best_std_error=True, N=0):
#     '''See https://arxiv.org/pdf/0712.0881.pdf p. 9 eq. 2.15 and 2.16

#     With regards to my notation :
#     X is D, the regressor/dictionary matrix
#     y is X, the measured signal we wish to reconstruct
#     beta is alpha, the coefficients
#     mu is the denoised reconstruction X_hat = D * alpha

#     likelihood formula have to assume normally distributed residuals from the reconstruction
#     though, since the clean signal intensity is impossible to estimate.
#     Besides, we only have a single point anyway to estimate from.
#     '''

#     y = y.ravel()
#     n = y.shape[0]
#     # k = X.shape[0]
#     p = X.shape[1]

#     if criterion == 'aic' or criterion == 'aicc':
#         w = 2
#     elif criterion == 'bic':
#         w = np.log(n)
#     elif criterion == 'ric':
#         w = 2 * np.log(p)
#     else:
#         raise ValueError('Criterion {} is not supported!'.format(criterion))

#     squared_error = (y[..., None] - mu)**2
#     mse = np.mean(squared_error, axis=0, dtype=np.float32)
#     # rss = np.sum(squared_error, axis=0, dtype=np.float32)
#     df_mu = np.sum(beta != 0, axis=0, dtype=np.float32)

#     # if variance is not None:
#     #     # likelihood for non normal distributions
#     #     if N >= 1:
#     #         pass # http://asa.scitation.org/doi/pdf/10.1121/1.400532
#     #         # we need an estimate of both sigma2 and eta though, but the whole game is to find eta in the first place...
#     #         # log_likelihood = np.log(y.sum() / variance * np.exp(-(np.sum(y**2) + mu**2) / (2*variance)) * i0(y.sum() * mu / variance))

#     #         # see if we can use the fixed point finder formulas for this part instead
#     #         # which would give us access to all these things in the first place
#     #     else:
#     #         # this is the likelihood for normal distributions though
#     #         log_likelihood = (-n/2) * np.log(2*np.pi) - (n/2) * np.log(variance) - rss / (2*variance)

#     # criterion according to 2.15 and 2.16 of https://projecteuclid.org/download/pdfview_1/euclid.aos/1194461726
#     variance = None
#     if variance is None:
#         variance = np.var(y)
#     # criterion_value = mse / variance + w/n * df_mu
#     criterion_value = n * np.log(mse) + w * df_mu

#     ## old criterion
#     # # Use mse = SSE/n estimate for sample variance - we assume normally distributed
#     # # residuals though for the log-likelihood function...
#     # if variance is None:
#     #     with np.errstate(divide='ignore'):
#     #         if n >= 2*p:
#     #             variance = rss / (n - df_mu - 1)
#     #         else:
#     #             variance = rss / (df_mu)
#     # # print(np.sqrt(variance), np.std(y), (np.sqrt(rss / (n - df_mu - 1))), n, p, df_mu.shape)
#     # log_likelihood = (-n/2) * np.log(2*np.pi) - (n/2) * np.log(variance) - rss / (2*variance)
#     # criterion_value = w * df_mu - 2 * log_likelihood


#     # variance = 100
#     # criterion_value = (rss / (n * variance)) + (df_mu * w / n)

#     #     # criterion_value = w * df_mu + n * np.log(rss)
#     #     # criterion_value = n * np.log(mse) + df_mu * w
#     #     criterion_value = (rss / (n * sigma2)) + (df_mu * w / n)
#     #     # criterion_value = n * np.log(rss) + df_mu * w
#     #     # print(n * np.log(mse), 'likelihood')
#     #     # print(df_mu * w, 'freedom')
#     #     # print('no variance')
#     #     # print('criterion', criterion_value)
#     #     # criterion_value = df_mu * w - 2 * np.log(rss)
#     #     # s2 = sse / n
#     #     # log_L = np.log(1 / np.sqrt(2 * np.pi * s2)) * n - sse / (2 * s2)
#     #     # criterion_value = w * df_mu - 2 * log_L
#     # else:
#     #     # criterion_value = (mse / variance) + (w * df_mu / n)
#     #     criterion_value = w * df_mu - 2 * log_likelihood
#     #     # print(w * df_mu, 'freedom')
#     #     # print(-2 * log_likelihood, '-likelihood')
#     #     # print('variance =', variance)
#     #     # print('criterion', criterion_value)
#     #     # criterion_value = rss / (n * variance) + (w * df_mu / n)
#     #     # aic = 2*df_mu - 2*np.log(mse)
#     #     # criterion_value = aic  + (2 * (df_mu + 1) * (df_mu + 2)) / (n - df_mu - 2)
#     #     # print(rss / (n * variance), 'rss')
#     #     # print(w, df_mu, n, 'addon')

#     if criterion == 'aicc':
#         with np.errstate(divide='ignore'):
#             aicc = (2 * df_mu * (df_mu + 1)) / (n - df_mu - 1)
#         # print(aicc, (2 * df_mu * (df_mu + 1)), (n - df_mu - 1))
#         criterion_value += aicc

#     # print(criterion_value, df_mu)
#     # 1/0
#     # We don't want empty models
#     criterion_value[df_mu == 0] = 1e300
#     criterion_value[np.isnan(criterion_value)] = 1e300
#     best_idx = np.argmin(criterion_value, axis=0)
#     # print(df_mu, df_mu[best_idx], best_idx)
#     # print((rss / (variance)) , (df_mu * w))
#     # print(criterion_value)
#     # print(criterion_value)
#     # print(df_mu)
#     # print(best_idx, len(criterion_value))
#     # print(rss)
#     # # We instead look for the best model + 1 std to be more robust
#     # if best_std_error:
#     #     std_mse = squared_error.std(axis=0, ddof=1)
#     #     # std_mse = np.sqrt(((mse - mean_mse)**2) / (y.shape[0] - 1))

#     #     se_mse = mse[best_idx] + std_mse[best_idx]
#     #     best_idx = np.argmin(se_mse <= criterion_value)
#     #     # min_mse = mean_mse.min()
#     #     # idx = mse <= mean_mse.min()
#     #     # idmin = np.max([idx])
#     #     # mse_min = mean_mse[idmin] + std_mse[idmin]
#     #     # best_idx = np.argmax(mean_mse <= mse_min)
#     #     # best_idx = np.argmin(criterion_value, axis=0)



#     #     # CVerr['lambda_min'] = scipy.amax(options['lambdau'][cvm <= scipy.amin(cvm)]).reshape([1])
#     #     # idmin = options['lambdau'] == CVerr['lambda_min']
#     #     # semin = cvm[idmin] + cvsd[idmin]
#     #     # CVerr['lambda_1se'] = scipy.amax(options['lambdau'][cvm <= semin]).reshape([1])
#     #     # CVerr['class'] = 'cvglmnet'

#     # print(criterion_value.shape, X.shape, y.shape, beta.shape, mu.shape, df_mu.shape, mse.shape, rss.shape, variance)
#     # print(best_idx, criterion_value.shape, criterion_value)
#     # print(best_idx, df_mu.shape, df_mu[best_idx]/X.shape[1], df_mu)

#     # We can only estimate sigma squared / residual error after selecting the best model
#     # if n > df_mu[best_idx]:
#     #     estimated_variance = np.sum((y - mu[:, best_idx])**2) / (n - df_mu[best_idx])
#     # else:
#     #     estimated_variance = 0

#     return best_idx


def lasso_path(X, y, nlam=100, fit_intercept=False, pos=False, standardize=False, weights=None, penalty=None, criterion=None):
    """return full path for Lasso"""

    # from glmnet_py import glmnet
    # X = np.array(X, copy=True, dtype=np.float64, order='F')
    # y = np.array(y, copy=True, dtype=np.float64, order='F').ravel()
    # fit = glmnet(x=X, y=y, family='gaussian', alpha=1., nlambda=nlam, intr=fit_intercept, standardize=standardize)
    # return fit['beta']

    # out = elastic_net_path(X, y, rho=1.0, nlam=nlam, fit_intercept=fit_intercept, pos=pos, standardize=standardize)
    # print('in path2', len(out))
    return elastic_net_path(X, y, rho=1.0, nlam=nlam, weights=weights, penalty=penalty, criterion=criterion,
                            fit_intercept=fit_intercept, pos=pos, standardize=standardize)


def elastic_net(X, y, rho, pos=False, thr=1e-7, weights=None, vp=None, copy=True, ulam=None, jd=np.zeros(1),
                standardize=False, nlam=100, maxit=100000, flmin=1e-4, fit_intercept=False, custom_path=False):
    """
    Raw-output wrapper for elastic net linear regression.
    X is D
    y is X
    rho for lasso/elastic net tradeoff
    """
    # custom_path = True
    # pos = False

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

    # this seems faster in our case so far though
    # ka = 1

    if custom_path:
        ulam = np.logspace(2, -5, num=nlam)

    # User activated values are used only if flmin >= 1.
    if ulam is not None:
        flmin = 2.

    # ne = int(0.25 * X.shape[1])
    # ne = float(0.25 * X.shape[1])
    # nx = float(0.95*X.shape[1])
    # nx = X.shape[1] + 1
    nvars = X.shape[1]
    ne = nvars + 1
    nx = int(min(ne * 2 + 20, X.shape[1]))
    # print(X.shape, y.shape, ne, nx)
    # print(weights.sum(), 'weights')
    # print(vp.sum(), 'vp')
    # 1/0
    # Call the Fortran wrapper.
    # flmin=1e-40
    # print(ulam, flmin)
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
    elif y.ndim == 2:
        nr = y.shape[1]
        # ne = min(X.shape[1], nx)
        isd = standardize
        jsd = 0  # standardize the response apparently
        intr = fit_intercept
        # maxit = maxit
        lmu = -1
        a0 = np.ones((nr, nlam), dtype=np.float64, order='F')
        ca = np.ones((nx, nr, nlam), dtype=np.float64, order='F')
        ia = np.ones(nx, dtype=np.float64, order='F')
        nin = np.ones(nlam, dtype=np.float64, order='F')
        rsq = np.ones(nlam, dtype=np.float64, order='F')
        alm = 0
        nlp = 0
        jerr = 0

        # actual call to function
        multelnet(rho,
                  X,
                  y,
                  weights,
                  jd,
                  vp,
                  box_constraints,
                  ne,
                  flmin,
                  ulam,
                  thr,
                  isd,
                  jsd,
                  intr,
                  maxit,
                  lmu,
                  a0,
                  ca,
                  ia,
                  nin,
                  rsq,
                  alm,
                  nlp,
                  jerr,
                  nlam=nlam,
                  nx=nx,
                  nr=nr)
    else:
        raise ValueError('y needs to be 1D or 2D but is {}'.format(y.ndim))

    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr


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


# This part stolen from
# https://github.com/ceholden/glmnet-python/blob/master/glmnet/utils.py


# def IC_path(X, y, coefs, intercepts, criterion='aic'):
#     """ Return AIC, BIC, or AICc for sets of estimated coefficients

#     Args:
#         X (np.ndarray): 2D (n_obs x n_features) design matrix
#         y (np.ndarray): 1D dependent variable
#         coefs (np.ndarray): 1 or 2D array of coefficients estimated from
#             GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
#         intercepts (np.ndarray): 1 or 2D array of intercepts from
#             GLMNET using one or more ``lambdas`` (n_lambdas)
#         criterion (str): AIC (Akaike Information Criterion), BIC (Bayesian
#             Information Criterion), or AICc (Akaike Information Criterion
#             corrected for finite sample sizes)

#     Returns:
#         np.ndarray: information criterion as 1D array (n_lambdas)

#     Note: AIC and BIC calculations taken from scikit-learn's LarsCV

#     """
#     # coefs = np.atleast_2d(coefs)

#     n_samples = y.shape[0]

#     criterion = criterion.lower()
#     if criterion == 'aic' or criterion == 'aicc':
#         K = 2
#     elif criterion == 'bic':
#         K = np.log(n_samples)
#     else:
#         raise ValueError('Criterion must be either AIC, BIC, or AICc')

#     residuals = y[:, np.newaxis] - (np.dot(X, coefs) + intercepts)
#     mse = np.mean(residuals**2, axis=0)
#     # df = np.zeros(coefs.shape[1], dtype=np.int16)
#     df = np.sum(coefs != 0, axis=-1)

#     # for k, coef in enumerate(coefs.T):
#     #     mask = np.abs(coef) > np.finfo(coef.dtype).eps
#     #     if not np.any(mask):
#     #         continue
#     #     df[k] = np.sum(mask)

#     with np.errstate(divide='ignore'):
#         criterion_ = n_samples * np.log(mse) + K * df
#         if criterion == 'aicc':
#             criterion_ += (2 * df * (df + 1)) / (n_samples - df - 1)

#     return criterion_
