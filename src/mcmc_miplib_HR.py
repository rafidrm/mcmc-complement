"""
We conduct MCMC complement sampling experiments in R^n using
random instances from MIPLIB.

Generating feasible points:
    - Too lazy to implement hitandrun, so I just randomly generate
    some vertices and then find convex combinations of them.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from pprint import pprint
import pulp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import sys
from tqdm import tqdm

import pudb


ATOL = 1e-5
RTOL = 1.0000001e-3
DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 1.00000001e-5
ALG_TIMEOUT_MULT = 12 


def lp_solver(c_vec, A_mat, b_vec):
    prob = pulp.LpProblem('prob', pulp.LpMinimize)
    m, n = A_mat.shape
    x = [pulp.LpVariable('x{}'.format(ix)) for ix in range(n)]
    prob += pulp.lpSum([ ci * xi for ci, xi in zip(c_vec, x) ])
    for ix in range(m):
        prob += pulp.lpSum([ ai * xi for ai, xi in zip(A_mat[ix], x) ]) >= b_vec[ix], 'con{}'.format(ix)
    prob.solve(pulp.GLPK(msg=0))
    if prob.status in [1, -2, -3]:
        x_vec = np.array([ xi.varValue for xi in x ])
        return x_vec
    else:
        return None


def hit_n_run_init(A_mat, b_vec):
    """
    Generate an initial decision in the interior of the feasible set.
    """
    _, n = A_mat.shape 
    vtx0 = lp_solver(
        np.zeros(n),
        A_mat,
        b_vec + np.random.exponential(scale=1e-1, size=len(b_vec))
    )
    vertices = [vtx0]
    for row in A_mat:
        vtx2 = lp_solver(
            row,
            A_mat,
            b_vec
        )
        if vtx2 is None:
            continue

        vertices.append(vtx2)
        init_pt = np.mean(np.array(vertices), axis=0)
        if is_bd(A_mat, b_vec, init_pt, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) == False:
            break
    if is_bd(A_mat, b_vec, init_pt, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) == True:
        raise Exception('Convex combination is still boundary')
    return init_pt


def hit_n_run(A_mat, b_vec, n_samples=200, hr_timeout=ALG_TIMEOUT_MULT):
    """ Hit and Run Sampler:

        1. Sample current point x
        2. Generate a random direction r
        3. Define gamma_i = ( b - a_i'x ) / ( r'a_i )
        4. Calculate max(gamma < 0) gamma_i and min(gamma > 0) gamma_i 
        5. Sample uniformly from [min_gamma, max_gamma]
    """
    m, n = A_mat.shape
    curr_pt = hit_n_run_init(A_mat, b_vec)
    pts = [curr_pt]
    pts_len = 1
    bar = tqdm(total=n_samples)
    for _ in range(n_samples * hr_timeout):
        direction = np.random.randn(n)
        direction = direction / np.linalg.norm(direction)
        # calculate gamma
        numer = b_vec - np.dot(A_mat, curr_pt)
        denom = np.dot(A_mat, direction)
        gamma = [ n / d for n, d in zip(numer, denom) ]
        gamma.append(0)
        gamma = np.array(gamma)
        if (gamma > 0).all():
            gamma_min = 0
        else:
            gamma_min = max(gamma[gamma < 0])
        if (gamma < 0).all():
            gamma_max = 0
        else:
            gamma_max = min(gamma[gamma > 0])
        magnitude = np.random.uniform(low=gamma_min, high=gamma_max)
        curr_pt = curr_pt + magnitude * direction
        if is_feasible(A_mat, b_vec, curr_pt):
            pts.append(curr_pt)
            bar.update(1)
            pts_len += 1
            if pts_len >= n_samples:
                break
        else:
            pass
    bar.close()
    if len(pts) < min(0.4 * n_samples, 500):
        raise Exception(
            'Sampled {} points instead of {}'.format(len(pts), 0.4 * n_samples)
        )
    return pts


def hit_n_run_infeas(A_mat, b_vec, A_hid, b_hid, n_samples=200, hr_timeout=ALG_TIMEOUT_MULT):
    """ Same as the hit and run sampler but now just checks to see if it is
    infeasible with respect to the hidden set.

        Hit and Run Sampler:

        1. Sample current point x
        2. Generate a random direction r and make sure that x - r is feasible
        3. Solve max { lambda | A (x - lambda * r) >= b }
        4. Sample uniformly from [0, lambda]
    """
    m, n = A_mat.shape
    curr_pt = hit_n_run_init(A_mat, b_vec)
    pts = [curr_pt]
    pts_len = 1
    bar = tqdm(total=n_samples)
    for _ in range(n_samples * hr_timeout):
    # for _ in tqdm(range(n_samples)):
        direction = np.random.randn(n)
        direction = direction / np.linalg.norm(direction)
        # calculate gamma
        numer = b_vec - np.dot(A_mat, curr_pt)
        denom = np.dot(A_mat, direction)
        gamma = [ n / d for n, d in zip(numer, denom) ]
        gamma.append(0)
        gamma = np.array(gamma)
        if (gamma > 0).all():
            gamma_min = 0
        else:
            gamma_min = max(gamma[gamma < 0])
        if (gamma < 0).all():
            gamma_max = 0
        else:
            gamma_max = min(gamma[gamma > 0])
        magnitude = np.random.uniform(low=gamma_min, high=gamma_max)
        curr_pt = curr_pt + magnitude * direction
        if is_feasible(A_mat, b_vec, curr_pt):
            if is_feasible(A_hid, b_hid, curr_pt) == False:
                pts.append(curr_pt)
                bar.update(1)
                pts_len += 1
                if pts_len >= n_samples:
                    break
    bar.close()
    if len(pts) < min(0.4 * n_samples, 500):
        raise Exception(
            'Sampled {} points instead of {}'.format(len(pts), 0.4 * n_samples)
        )
    return pts


def direction_sample_helper(cons):
    """ Samples a random point from the unit ball then checks with
    the a vector that con'pt > 0
    """
    wrong_direction = 1
    n = len(cons[0])
    while wrong_direction > 0:
        pt = np.random.randn(n)
        pt = pt / np.linalg.norm(pt)
        if all([np.dot(con, pt) >= 0 for con in cons]):
            wrong_direction = 0
        else:
            wrong_direction += 1
            if wrong_direction > 1e12:
                raise Exception('Direction sampler stuck.')
    return pt


def direction_sample(A_mat, b_vec, bd_pt):
    """ 
    First identifies the relevant constraint on the boundary,
    then calls sample helper.
    """
    delta = np.abs(np.dot(A_mat, bd_pt) - b_vec)
    cons = [ A_mat[ix] for ix in range(len(delta)) if delta[ix] < ATOL ]
    assert len(cons) > 0, 'This is not a boundary point.'
    return direction_sample_helper(cons)


def get_next_bd_pt(A_mat, b_vec, bd_pt, dir_pt):
    """ 
    First removes boundary constraints, then finds nearest boundary.
    """
    weights = np.array([(bi - np.dot(ai, bd_pt)) / np.dot(ai, dir_pt) for ai, bi in zip(A_mat, b_vec)])
    weights[weights <= 0] = 1e12
    weight = min(weights)
    tmp = bd_pt + weight * dir_pt
    if is_bd(A_mat, b_vec, tmp) == False:
        print('not bd.')

    return bd_pt + weight * dir_pt


def shuffle_A_b(A_mat, b_vec):
    """
    Shuffle the rows of the A matrix and b vector accordingly.
    """
    m, n = A_mat.shape
    A_new = np.zeros((m, n))
    b_new = np.zeros(m)
    reorder = np.random.permutation(list(range(m)))
    for i, j in enumerate(reorder):
        A_new[i] = A_mat[j]
        b_new[i] = b_vec[j]
    return A_new, b_new


def hr_test_set(A_mat, b_vec, A_relax, b_relax, n_samples):
    """
    Generate both test sets using hit and run.
    """
    A_mat, b_vec = shuffle_A_b(A_mat, b_vec)
    A_relax, b_relax = shuffle_A_b(A_relax, b_relax)
    feas_pts = hit_n_run(A_mat, b_vec, n_samples=n_samples)
    feas_pts = np.array(feas_pts)

    infeas_pts = hit_n_run_infeas(A_relax, b_relax, A_mat, b_vec, n_samples)
    infeas_pts = np.array(infeas_pts)

    num_pts = min(len(feas_pts), len(infeas_pts))
    feas_pts = feas_pts[:num_pts]
    infeas_pts = infeas_pts[:num_pts]

    X_test = np.vstack([feas_pts, infeas_pts])
    y_test = []
    for xi in X_test:
        y_test.append(int(is_feasible(A_mat, b_vec, xi)))
    print(
        'Generated {} feasible and {} infeasible pts for testing.'.format(
            np.sum(y_test),
            len(y_test) - np.sum(y_test),
        )
    )
    return X_test, np.array(y_test)


def shake_n_bake_infeas(A_mat, b_vec, init_pt, n_samples=10, scale=1, hid=[0,0], sb_timeout=ALG_TIMEOUT_MULT):
    """
    1. randomly sample direction vector (r)
    2. randomly sample magnitude (xi)
    3. add infeasible point (y - xi * r, y)
    4. get next boundary point
    
    Use the hid feasible set if we run into numerical issues with the scale
    parameter. Then, feasibility is measured using the hidden set which must
    be satisfied, so even if it is "infeasible" by a small amount on the real
    feasible set, we are okay.
    """
    if hid == [0, 0]:
        A_hid, b_hid = A_mat, b_vec
    else:
        A_hid, b_hid = hid[0], hid[1]
    dataset = []
    dataset_len = 0
    bd_pt = init_pt
    bar = tqdm(total=n_samples)
    for ix in range(n_samples * sb_timeout):
        r = direction_sample(A_mat, b_vec, bd_pt)
        xi = np.random.exponential(scale=scale)
        infeas_pt = bd_pt - xi * r
        if is_feasible(A_hid, b_hid, infeas_pt) == False:
            dataset.append(infeas_pt)
            assert is_bd(A_mat, b_vec, bd_pt), 'sb using not bd.'
            bar.update(1)
            dataset_len += 1
            if dataset_len >= n_samples:
                break
        else:
            pass
        bd_pt = get_next_bd_pt(A_mat, b_vec, bd_pt, r)
    bar.close()
    if len(dataset) < min(0.4 * n_samples, 500):
        raise Exception(
            'Sampled {} points instead of {}'.format(len(dataset), 0.4 * n_samples)
        )
    return np.array(dataset)


def shake_n_bake_safe_init(A_mat, b_vec, feas_pt):
    """
    Projects a feasible point to a random facet.
    """
    m, n = A_mat.shape
    ix = np.random.choice(m)
    a_row = A_mat[ix] + np.random.randn(n) * RTOL
    bd_pt = get_next_bd_pt(A_mat, b_vec, feas_pt, a_row)
    assert is_bd(A_mat, b_vec, bd_pt), 'SB init is not a boundary point'
    return bd_pt


def is_feasible(A_mat, b_vec, pt, tol=1e-5):
    # Assuming inequality form
    true_tol = np.linalg.norm(b_vec) * tol
    if (np.dot(A_mat, pt) >= b_vec - true_tol).all():
        return True
    else:
        return False


def is_bd(A_mat, b_vec, pt, rtol=RTOL, atol=ATOL):
    # Assuming inequality form
    if is_feasible(A_mat, b_vec, pt) == False:
        return False
    if np.isclose(np.dot(A_mat, pt), b_vec, rtol=rtol, atol=atol).any():
        return True
    else:
        return False


def label_feas(A_mat, b_vec, pt, tol=1e-5):
    m = len(b_vec)
    true_tol = np.linalg.norm(b_vec) * tol
    if is_feasible(A_mat, b_vec, pt):
        label = m
    else:
        labels = []
        for ix in range(m):
            if np.dot(A_mat[ix], pt) < b_vec[ix] - true_tol:
                labels.append(ix)
        label = np.random.choice(labels)
    return label


def train_and_test_gmm(
        feas_pts,
        X_test,
        y_test,
        n_components=0.95,
        use_pca=True,
        verbose=True
    ):
    if use_pca == True:
        pca = PCA(n_components)
        X_train = pca.fit_transform(feas_pts)
        X_test = pca.transform(X_test)
    else:
        X_train = feas_pts

    params = {
        'n_components': [ 1, 5, 10, 50, 100]
    }
    grid = GridSearchCV(GaussianMixture(), params, cv=3)
    grid.fit(X_train)
    gmm = grid.best_estimator_
    gmm.thresh = np.min(np.exp(gmm.score_samples(X_train)))
    y_pred = np.exp(gmm.score_samples(X_test))

    y_pred[y_pred > gmm.thresh] = 1
    y_pred[y_pred < gmm.thresh] = 0
    y_pred[y_pred == gmm.thresh] = 1

    acc = 1 - np.mean(np.abs(y_test - y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    if verbose == True:
        print('GMM Test set score = {}'.format(acc))
        print('TP = {} FP = {}'.format(tp, fp))
        print('FN = {} TN = {}'.format(fn, tn))

    res = {
        'n_components': gmm.n_components,
        'thresh': gmm.thresh,
        'acc': np.round(acc, 3),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fpr': np.round(fp / (tn + fp + DEFAULT_ATOL), 3),
        'tpr': np.round(tp / (tp + fn + DEFAULT_ATOL), 3),
        'pre': np.round(tp / (tp + fp + DEFAULT_ATOL), 3),
    }
    return res


def train_and_test_kde(
        feas_pts,
        X_test,
        y_test,
        n_components=0.95,
        use_pca=True,
        verbose=True
    ):
    if use_pca == True:
        pca = PCA(n_components)
        X_train = pca.fit_transform(feas_pts)
        X_test = pca.transform(X_test)
    else:
        X_train = feas_pts

    params = {
        'bandwidth': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    }
    grid = GridSearchCV(KernelDensity(), params, cv=3)
    grid.fit(X_train)
    kde = grid.best_estimator_
    kde.thresh = np.min(np.exp(kde.score_samples(X_train)))
    y_pred = np.exp(kde.score_samples(X_test))

    y_pred[y_pred > kde.thresh] = 1
    y_pred[y_pred < kde.thresh] = 0
    y_pred[y_pred == kde.thresh] = 1

    acc = 1 - np.mean(np.abs(y_test - y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    if verbose == True:
        print('KDE Test set score = {}'.format(acc))
        print('TP = {} FP = {}'.format(tp, fp))
        print('FN = {} TN = {}'.format(fn, tn))

    res = {
        'bw': kde.bandwidth,
        'thresh': kde.thresh,
        'acc': np.round(acc, 3),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fpr': np.round(fp / (tn + fp + DEFAULT_ATOL), 3),
        'tpr': np.round(tp / (tp + fn + DEFAULT_ATOL), 3),
        'pre': np.round(tp / (tp + fp + DEFAULT_ATOL), 3),
    }
    return res


def train_and_test_classifier(
        feas_pts,
        infeas_pts,
        X_test,
        y_test,
        model=GradientBoostingClassifier,
        use_pca=False,
        n_components=0.99,
        verbose=True,
    ):
    X_train = np.vstack((feas_pts, infeas_pts))
    feas_labels = np.array([np.ones(len(feas_pts))]).T
    infeas_labels = np.array([np.zeros(len(infeas_pts))]).T
    y_train = np.vstack(
        (feas_labels, infeas_labels)
    ).T[0]
    if use_pca == True:
        pca = PCA(n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    clf = model()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
    
    if verbose == True:
        print('MDL Test set score = {}'.format(acc))
        print('TP = {} FP = {}'.format(tp, fp))
        print('FN = {} TN = {}'.format(fn, tn))

    res = {
        'acc': np.round(acc, 3),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fpr': np.round(fp / (tn + fp + DEFAULT_ATOL), 3),
        'tpr': np.round(tp / (tp + fn + DEFAULT_ATOL), 3),
        'pre': np.round(tp / (tp + fp + DEFAULT_ATOL), 3),
    }
    return res


def experiment(fname, params):
    n_samples = params.get('n_samples', 200)
    pca_n_components = params.get('pca_n_components', 0.95)
    scale = params.get('scale', 1)
    train_scale = params.get('train_scale', 0.1)
    test_n_samples = params.get('test_n_samples', 200)
    use_pca = params.get('use_pca', False)
    verbose = params.get('verbose', True)
    if verbose:
        pprint(params)

    data = pickle.load(open(fname, 'rb'))
    c_vec = data['c_vec']
    A_mat, b_vec = data['A_mat'], data['b_vec']
    A_relax, b_relax = data['A_relax'], data['b_relax']

    feas_pts = hit_n_run(A_mat, b_vec, n_samples=n_samples)
    feas_pts = np.array(feas_pts)
    feas_check = [ is_feasible(A_mat, b_vec, pt) == True for pt in feas_pts ]
    assert all(feas_check), 'Generated an infeasible point'
    n_samples = len(feas_pts)

    init_pt = shake_n_bake_safe_init(A_relax, b_relax, feas_pts[-1])
    mcmc_infeas_pts = shake_n_bake_infeas(
        A_relax,
        b_relax,
        init_pt,
        n_samples=n_samples,
        scale=train_scale,
    )
    feas_check = [ is_feasible(A_mat, b_vec, pt) == False for pt in mcmc_infeas_pts ]
    assert all(feas_check), 'Generated feasible point instead of infeasible'

    # cut to make sure we have the same training set for everything
    n_samples = min(len(feas_pts), len(mcmc_infeas_pts))
    feas_pts = feas_pts[:n_samples]
    mcmc_infeas_pts = mcmc_infeas_pts[:n_samples]
    print(
        'Generated {} feasible and {} infeasible for training.'.format(
            len(feas_pts),
            len(mcmc_infeas_pts)
        )
    )

    X_test, y_test = hr_test_set(
        A_mat,
        b_vec,
        A_relax,
        b_relax,
        test_n_samples,
    )
    kde_res = train_and_test_kde(
        feas_pts,
        X_test,
        y_test,
        n_components=pca_n_components,
        use_pca=use_pca,
        verbose=verbose
    )
    gmm_res = train_and_test_gmm(
        feas_pts,
        X_test,
        y_test,
        n_components=pca_n_components,
        use_pca=use_pca,
        verbose=verbose
    )
    mcmc_res = train_and_test_classifier(
        feas_pts,
        mcmc_infeas_pts,
        X_test,
        y_test,
        model=GradientBoostingClassifier,
        use_pca=use_pca,
        n_components=pca_n_components,
        verbose=verbose,
    )
    stats = {
        'training_n_samples': len(feas_pts) + len(mcmc_infeas_pts),
        'testing_n_samples': len(y_test),
    }
    return kde_res, mcmc_res, gmm_res, stats


def experiment_n_tries(fname, params):
    params = parse_fname(fname, params)
    MAX_RUNS = params.get('MAX_RUNS', 50)
    n_runs = params.get('n_runs', 1)

    _run = [0, 0]  # number of success, number of runs
    all_kde_res = []
    all_mcmc_res = []
    all_gmm_res = []
    while _run[0] < n_runs:
        try:
            kde_res, mcmc_res, gmm_res, stats = experiment(fname, params)
            all_kde_res.append(kde_res)
            all_mcmc_res.append(mcmc_res)
            all_gmm_res.append(gmm_res)
            _run[0] += 1
            print(
                'Run {} - \tScores:\tGMM:{}\tKDE:{}\tMCMC:{}'.format(
                    _run,
                    gmm_res['acc'],
                    kde_res['acc'],
                    mcmc_res['acc']
                )
            )
        except Exception as inst:
            print(inst)
            _run[1] += 1

            if _run[1] > MAX_RUNS:
                print('\tCould only run up to {}'.format(_run))
                _run[0] = n_runs

    res = {
        'kde_acc': np.mean([kde['acc'] for kde in all_kde_res]),
        'kde_tn': np.mean([kde['tn'] for kde in all_kde_res]),
        'kde_tp': np.mean([kde['tp'] for kde in all_kde_res]),
        'kde_fn': np.mean([kde['fn'] for kde in all_kde_res]),
        'kde_fp': np.mean([kde['fp'] for kde in all_kde_res]),
        'kde_fpr': np.mean([kde['fpr'] for kde in all_kde_res]),
        'kde_tpr': np.mean([kde['tpr'] for kde in all_kde_res]),
        'kde_pre': np.mean([kde['pre'] for kde in all_kde_res]),
        'mcmc_acc': np.mean([mcmc['acc'] for mcmc in all_mcmc_res]),
        'mcmc_tn': np.mean([mcmc['tn'] for mcmc in all_mcmc_res]),
        'mcmc_tp': np.mean([mcmc['tp'] for mcmc in all_mcmc_res]),
        'mcmc_fn': np.mean([mcmc['fn'] for mcmc in all_mcmc_res]),
        'mcmc_fp': np.mean([mcmc['fp'] for mcmc in all_mcmc_res]),
        'mcmc_fpr': np.mean([mcmc['fpr'] for mcmc in all_mcmc_res]),
        'mcmc_tpr': np.mean([mcmc['tpr'] for mcmc in all_mcmc_res]),
        'mcmc_pre': np.mean([mcmc['pre'] for mcmc in all_mcmc_res]),
        'gmm_acc': np.mean([gmm['acc'] for gmm in all_gmm_res]),
        'gmm_tn': np.mean([gmm['tn'] for gmm in all_gmm_res]),
        'gmm_tp': np.mean([gmm['tp'] for gmm in all_gmm_res]),
        'gmm_fn': np.mean([gmm['fn'] for gmm in all_gmm_res]),
        'gmm_fp': np.mean([gmm['fp'] for gmm in all_gmm_res]),
        'gmm_fpr': np.mean([gmm['fpr'] for gmm in all_gmm_res]),
        'gmm_tpr': np.mean([gmm['tpr'] for gmm in all_gmm_res]),
        'gmm_pre': np.mean([gmm['pre'] for gmm in all_gmm_res]),
    }
    summary = res.copy()
    summary.update(params)
    pprint(summary)
    return summary


def parse_fname(fname, params):
    tokens = fname.name.split('_')
    params['prob'] = tokens[0]
    params['prob_scale'] = float(tokens[-2][5:])
    params['prob_round'] = int(tokens[-1][1:].split('.')[0])
    return params


def clean_df(df):
    df = df.T
    COLS = [
        'kde_acc', 'kde_tn', 'kde_tp', 'kde_fn', 'kde_fp', 'kde_fpr',
        'kde_tpr', 'kde_pre', 'mcmc_acc', 'mcmc_tn', 'mcmc_tp', 'mcmc_fn',
        'mcmc_fp', 'mcmc_fpr', 'mcmc_tpr', 'mcmc_pre', 'gmm_acc', 'gmm_tn',
        'gmm_tp', 'gmm_fn', 'gmm_fp', 'gmm_fpr', 'gmm_tpr', 'gmm_pre',
    ]
    df2 = df[COLS + ['prob']].copy()
    df2[COLS] = df2[COLS].astype('float')
    df2 = df2.groupby('prob').mean().T
    return df2


def sweep_tests(p, r, params):
    files_to_ignore = [
        Path('../miplib/neos16_R1.pickle'),
        Path('../miplib/markshare_4_0_R1.pickle'),
        Path('../miplib/timtab1CUTS_R1.pickle'), # cannot start hitnrun
    ]

    all_summaries = {}
    
    counter = 0
    for fname in p.iterdir():
        if fname in files_to_ignore:
            continue
        # if fname != Path('../miplib2/gen-ip021_R5.pickle'):
        #     continue
        if fname.suffix == '.pickle':
            counter += 1
            print('\n'+('*'*80)+'\n'+('*'*80))
            print(fname)
            print('\t({})'.format(counter))
            print('*'*80)
            print('*'*80)
            summary = experiment_n_tries(fname, params)
            all_summaries[fname.name] = summary
            params_final = parse_fname(fname, params)

    df = pd.DataFrame.from_dict(all_summaries)
    df_summary = clean_df(df)
    rname = 'TMP_HR_sweep_tests_{}_{}'.format(
        params_final['prob'],
        np.round(np.random.rand(), 5)
    )
    df.to_csv( r / (rname + '.csv') )
    df_summary.to_csv( r / (rname + '_summary.csv') )
    pprint(df_summary)


if __name__ == "__main__":
    p = Path('../enlight8_v2/')
    r = Path('../results/mar19')

    params = {
        'MAX_RUNS': 100,
        'n_samples': 16000,
        # 'n_samples': 4000,
        'n_runs': 1,
        'pca_n_components': 1,
        # 'pca_n_components': 0.5,
        'scale': 1,
        'train_scale': 1,
        'test_n_samples': 16000,
        # 'test_n_samples': 4000,
        'use_pca': False,
        # 'use_pca': True,
        'verbose': False,
    }
    sweep_tests(p, r, params)
