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
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

import pudb


ATOL = 1e-5
RTOL = 1.0000001e-3
DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 1.00000001e-5


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


def hit_n_run_init(c_vec, A_mat, b_vec):
    """
    Generate an initial decision in the interior of the feasible set.
    """
    n = len(c_vec)
    vtx0 = lp_solver(
        np.zeros(n),
        A_mat,
        b_vec + np.random.exponential(scale=1e-1)
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

        for vtx in vertices:
            if np.isclose(vtx, vtx2, rtol=RTOL, atol=ATOL).all() == True:
                continue

        vertices.append(vtx2)
        init_pt = np.mean(np.array(vertices), axis=0)
        if is_bd(A_mat, b_vec, init_pt, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) == True:
            continue
        else:
            break
    if is_bd(A_mat, b_vec, init_pt, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) == True:
        raise Exception('Convex combination is still boundary')
    return init_pt


def hit_n_run(c_vec, A_mat, b_vec, n_samples=200):
    """ Hit and Run Sampler:

        1. Sample current point x
        2. Generate a random direction r and make sure that x - r is feasible
        3. Solve max { lambda | A (x - lambda * r) >= b }
        4. Sample uniformly from [0, lambda]
    """
    m, n = A_mat.shape
    curr_pt = hit_n_run_init(c_vec, A_mat, b_vec)
    pts = [curr_pt]
    for _ in tqdm(range(n_samples)):
        direction = ( np.random.rand(n) - 0.5 )
        # push it a little bit to be feasible
        A_x = np.dot(A_mat, curr_pt) - b_vec 
        A_direction = np.dot(A_mat, direction)
        search_ub = pulp.LpProblem('search_ub', pulp.LpMaximize)
        gamma = pulp.LpVariable('gamma')
        # upper bound of search range
        search_ub += gamma
        for ix in range(m):
            search_ub += gamma * A_direction[ix] <= A_x[ix]
        search_ub.solve(pulp.GLPK(msg=0))
        if search_ub.status == 1:
            max_magnitude = gamma.varValue * 0.99
        elif search_ub.status in [-2, -3]:
            max_magnitude = 1e16
        else:
            max_magnitude = 0
        # lower bound of search range
        search_lb = pulp.LpProblem('search_lb', pulp.LpMinimize)
        gamma = pulp.LpVariable('gamma')
        search_lb += gamma
        for ix in range(m):
            search_lb += gamma * A_direction[ix] <= A_x[ix]
        search_lb.solve(pulp.GLPK(msg=0))
        if search_lb.status == 1:
            min_magnitude = gamma.varValue * 0.99
        elif search_lb.status in [-2, -3]:
            min_magnitude = -1e16
        else:
            min_magnitude = 0
        magnitude = np.random.uniform(low=min_magnitude, high=max_magnitude)
        curr_pt = curr_pt - magnitude * direction
        if is_feasible(A_mat, b_vec, curr_pt):
            pts.append(curr_pt)
            # if len(pts) > 0.9 * n_samples:
            #     break
    if len(pts) < min(0.5 * n_samples, 500):
        raise Exception(
            'Sampled {} points instead of {}'.format(len(pts), 0.5 * n_samples)
        )
    print('Sampled {} points for HR'.format(len(pts)))
    return pts


def direction_sample_helper(cons):
    """ Samples a random point from the unit ball then checks with
    the a vector that con'pt > 0
    """
    wrong_direction = 1
    n = len(cons[0])
    while wrong_direction > 0:
        pt = np.random.rand(n) - 0.5
        pt = pt / np.linalg.norm(pt)
        if all([np.dot(con, pt) >= 0 for con in cons]):
            wrong_direction = 0
        else:
            wrong_direction += 1
            if wrong_direction > 1e12:
                raise Exception('Direction sampler stuck.')
    return pt


def direction_sample(A_mat, b_vec, bd_pt):
    """ First identifies the relevant constraint on the boundary,
    then calls sample helper.
    """
    delta = np.abs(np.dot(A_mat, bd_pt) - b_vec)
    cons = [ A_mat[ix] for ix in range(len(delta)) if delta[ix] < ATOL ]
    assert len(cons) > 0, 'This is not a boundary point.'
    #ind = np.argmin(delta)
    #val = np.min(delta)
    #if val > ATOL:
    #    print('No boundary, gap is {}'.format(val))
    #con = A_mat[ind]
    return direction_sample_helper(cons)


def get_next_bd_pt(A_mat, b_vec, bd_pt, dir_pt):
    """ First removes boundary constraints, then finds nearest
    boundary.
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
    

def mixed_test_set(c_vec, A_mat, b_vec, n_samples, scale=1):
    """
    Will generate feasible points using hitnrun.
    and infeasible points using shakenbake and projection.
    """
    A_mat, b_vec = shuffle_A_b(A_mat, b_vec)
    feas_n_samples = int(n_samples / 2)
    proj_n_samples = int(n_samples / 4)
    mcmc_n_samples = int(n_samples / 4) 
    feas_pts = hit_n_run(c_vec, A_mat, b_vec, n_samples=feas_n_samples)
    feas_pts = np.array(feas_pts)

    proj_infeas_pts = constraint_projection_sampler(
        A_mat,
        b_vec,
        feas_pts,
        n_samples=proj_n_samples,
        scale=scale,
    )

    # init_pt = shake_n_bake_init(A_mat[0], A_mat, b_vec)
    init_pt = shake_n_bake_init(c_vec, A_mat, b_vec)
    mcmc_infeas_pts = shake_n_bake_infeas(
        A_mat,
        b_vec,
        init_pt,
        n_samples=mcmc_n_samples,
        scale=scale
    )
    
    X_test = np.vstack([feas_pts, proj_infeas_pts, mcmc_infeas_pts])
    y_test = []
    for xi in X_test:
        y_test.append(int(is_feasible(A_mat, b_vec, xi)))
    print(
        'Generated {} feasible and {} ({}, {}) infeasible pts for testing.'.format(
            np.sum(y_test),
            len(y_test) - np.sum(y_test),
            len(proj_infeas_pts),
            len(mcmc_infeas_pts),
        )
    )
    return X_test, np.array(y_test)


def mcmc_test_set(c_vec, A_mat, b_vec, n_samples, scale=1):
    """
    Will generate feasible points using hitnrun.
    and infeasible points using shakenbake and projection.
    """
    A_mat, b_vec = shuffle_A_b(A_mat, b_vec)
    feas_n_samples = int(n_samples / 2)
    mcmc_n_samples = int(n_samples / 2) 
    feas_pts = hit_n_run(c_vec, A_mat, b_vec, n_samples=feas_n_samples)
    feas_pts = np.array(feas_pts)

    # init_pt = shake_n_bake_init(A_mat[0], A_mat, b_vec)
    init_pt = shake_n_bake_init(c_vec, A_mat, b_vec)
    mcmc_infeas_pts = shake_n_bake_infeas(
        A_mat,
        b_vec,
        init_pt,
        n_samples=mcmc_n_samples,
        scale=scale
    )
    
    X_test = np.vstack([feas_pts, mcmc_infeas_pts])
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



def shake_n_bake_infeas(A_mat, b_vec, init_pt, n_samples=10, scale=1, b_hid=[0]):
    """
    1. randomly sample direction vector (r)
    2. randomly sample magnitude (xi)
    3. add infeasible point (y - xi * r, y)
    4. get next boundary point
    """
    if all(b_hid) == 0:
        b_hid = b_vec
    dataset = []
    bd_pt = init_pt
    for ix in tqdm(range(n_samples)):
        r = direction_sample(A_mat, b_vec, bd_pt)
        xi = np.random.exponential(scale=scale)
        infeas_pt = bd_pt - xi * r
        if is_feasible(A_mat, b_hid, infeas_pt) == False:
            dataset.append(infeas_pt)
            assert is_bd(A_mat, b_vec, bd_pt), 'sb using not bd.'
        else:
            print('shake n bake found feasible pt.')
            # pass
        bd_pt = get_next_bd_pt(A_mat, b_vec, bd_pt, r)
    return np.array(dataset)


def shake_n_bake_init(c_vec, A_mat, b_vec):
    m, n = A_mat.shape
    init_pt = lp_solver(c_vec, A_mat, b_vec)
    if init_pt is None:
        raise Exception('Could not initialize shake_n_bake')
    return init_pt


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


def haussdorf(feas, infeas, inner=np.mean, outer=np.mean):
    inner_dist = np.zeros(len(feas))
    for ix in range(len(inner_dist)):
        inner_dist[ix] = inner([np.linalg.norm(feas[ix] - iy) for iy in infeas])
    return outer(inner_dist)


def constraint_projection_sampler(A_mat, b_vec, feas_pts, n_samples=200, scale=1):
    m, n = A_mat.shape
    infeas_pts = []
    for ix in tqdm(range(n_samples)):
        idx = np.random.randint(0, len(feas_pts))
        pt = feas_pts[idx]
        row_idx = np.random.randint(0, m)
        ai, bi = A_mat[row_idx], b_vec[row_idx]
        xi_lb = ( bi - np.dot(ai, pt) ) / ( np.dot(ai, ai) ) 
        xi = xi_lb - np.random.exponential(scale=scale)
        if is_feasible(A_mat, b_vec, pt + xi * ai) == False:
            infeas_pts.append(pt + xi * ai)
    if len(infeas_pts) < 0.9 * n_samples:
        raise Exception('Constraint projection did not find enough points.')
    return np.array(infeas_pts)


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


def experiment_MCMC(
        fname, 
        n_samples=200,
        randomizer=np.random.rand,
        test_size=0.5,
        scale=1,
        multiclass=True,
    ):
    data = pickle.load(open(fname, 'rb'))
    c_vec = data['c_vec']
    A_mat, b_vec, b_relax = data['A_mat'], data['b_vec'], data['b_relax']
    print(A_mat.shape)
    feas_pts = hit_n_run(c_vec, A_mat, b_vec, n_samples=n_samples)
    feas_pts = np.array(feas_pts)
    feas_check = [ is_feasible(A_mat, b_vec, pt) == True for pt in feas_pts ]
    assert all(feas_check), 'Generated infeasible point instead of feasible'
    n_samples = len(feas_pts)

    print('Running projection sampler ...')
    proj_infeas_pts = constraint_projection_sampler(
        A_mat,
        b_relax,
        feas_pts,
        n_samples=n_samples,
        scale=(scale),
    )
    feas_check = [ is_feasible(A_mat, b_vec, pt) == False for pt in proj_infeas_pts ]
    # feas_check = [ is_feasible(A_mat, b_relax, pt) == False for pt in proj_infeas_pts ]
    assert all(feas_check), 'Generated feasible point instead of infeasible'
    print('Projected {} points'.format(len(proj_infeas_pts)))

    print('Running SB sampler ...')
    init_pt = shake_n_bake_init(c_vec, A_mat, b_relax)
    mcmc_infeas_pts = shake_n_bake_infeas(
        A_mat,
        b_relax,
        init_pt,
        n_samples=n_samples,
        scale=(scale),
        b_hid=b_vec,
    )
    feas_check = [ is_feasible(A_mat, b_vec, pt) == False for pt in mcmc_infeas_pts ]
    # feas_check = [ is_feasible(A_mat, b_relax, pt) == False for pt in mcmc_infeas_pts ]
    assert all(feas_check), 'Generated feasible point instead of infeasible'
    print('MCMC {} points'.format(len(mcmc_infeas_pts)))
    
    # dist_mcmc = haussdorf(feas_pts, mcmc_infeas_pts, inner=np.min, outer=np.min) 
    # dist_proj = haussdorf(feas_pts, proj_infeas_pts, inner=np.min, outer=np.min) 
    # print('MCMC {} PROJ {}'.format(dist_mcmc, dist_proj))

    # Evaluate models
    print('Generating testing data ... ')
    X_test, y_test = mcmc_test_set(c_vec, A_mat, b_vec, n_samples, scale=scale)
    # X_test, y_test = mixed_test_set(c_vec, A_mat, b_vec, n_samples, scale=scale)
    n_feas = len(feas_pts)
    n_infeas = min(len(proj_infeas_pts), len(mcmc_infeas_pts))
    n_pts = min(n_feas, n_infeas)
    feas_pts = feas_pts[:n_pts]
    proj_infeas_pts = proj_infeas_pts[:n_pts]
    mcmc_infeas_pts = mcmc_infeas_pts[:n_pts]
    print('Training and testing models ...')
    if multiclass == True:
        proj_res = train_and_test_multiclassifier(
            feas_pts,
            proj_infeas_pts,
            X_test,
            A_mat,
            b_vec,
            model=LinearSVC
        )
        mcmc_res = train_and_test_multiclassifier(
            feas_pts,
            mcmc_infeas_pts,
            X_test,
            A_mat,
            b_vec,
            model=LinearSVC
        )
    else:
        proj_res = train_and_test_classifier(
            feas_pts,
            proj_infeas_pts,
            X_test=X_test,
            y_test=y_test,
            model=GradientBoostingClassifier,
            # model=AdaBoostClassifier,
        )
        mcmc_res = train_and_test_classifier(
            feas_pts,
            mcmc_infeas_pts,
            X_test=X_test,
            y_test=y_test,
            model=GradientBoostingClassifier,
            # model=AdaBoostClassifier,
        )

    gaus_res = train_and_test_kde(
    # gaus_acc = train_and_test_gmm(
        feas_pts,
        X_test,
        y_test,
    )

    summary = {
        'proj_res': proj_res,
        'mcmc_res': mcmc_res,
        'gaus_res': gaus_res,
    }
    pprint(summary)
    return summary


def train_and_test_multiclassifier(
        feas_pts,
        infeas_pts,
        X_test,
        A_mat,
        b_vec,
        model=SVC,
    ):
    X_train = np.vstack((feas_pts, infeas_pts))
    y_train = np.array(
        [label_feas(A_mat, b_vec, pt) for pt in X_train]
    )
    y_test = np.array(
        [label_feas(A_mat, b_vec, pt) for pt in X_test]
    )
    clf = model(class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # pu.db

    m = len(b_vec)
    y_test[y_test < m] = 0
    y_test[y_test == m] = 1
    y_pred[y_pred < m] = 0
    y_pred[y_pred == m] = 1
    acc = 1 - np.mean(np.abs(y_test - y_pred))
    print('MDL Test set score = {}'.format(acc))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('TP = {} FP = {}'.format(tp, fp))
    print('FN = {} TN = {}'.format(fn, tn))

    res = {
        'acc': acc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'prec': tp / (tp + fp),
        'rec': tp / (tp + fn),
    }
    return res


def train_and_test_classifier(
        feas_pts,
        infeas_pts,
        X_test,
        y_test,
        model=SVC,
    ):
    X_train = np.vstack((feas_pts, infeas_pts))
    feas_labels = np.array([np.ones(len(feas_pts))]).T
    infeas_labels = np.array([np.zeros(len(infeas_pts))]).T
    y_train = np.vstack(
        (feas_labels, infeas_labels)
    ).T[0]
    clf = model()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print('MDL Test set score = {}'.format(acc))
    tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
    print('TP = {} FP = {}'.format(tp, fp))
    print('FN = {} TN = {}'.format(fn, tn))

    res = {
        'acc': acc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'prec': tp / (tp + fp + DEFAULT_ATOL),
        'rec': tp / (tp + fn + DEFAULT_ATOL),
    }
    return res


def train_and_test_gmm(feas_pts, X_test, y_test):
    # n_components = [ 5 ]
    n_components = [ 1, 5, 10, 20, 50, 100, 500]
    max_acc = 0
    for comp in n_components:
        # train gmm
        mdl = GaussianMixture(n_components=comp)
        try:
            mdl.fit(feas_pts)
            # pu.db
            min_score = np.min(-1 / mdl.score_samples(feas_pts))
            # test classifier
            y_pred = -1 / mdl.score_samples(X_test)
            y_pred[y_pred > min_score] = 1
            y_pred[y_pred <= min_score] = 0

            acc = 1 - np.mean(np.abs(y_test - y_pred))
            print('Component {} \t Test set score = {}'.format(comp, acc))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print('TP = {} FP = {}'.format(tp, fp))
            print('FN = {} TN = {}'.format(fn, tn))
            if acc > max_acc:
                res = {
                    'acc': acc,
                    'tn': tn,
                    'fp': fp,
                    'fn': fn,
                    'tp': tp,
                    'prec': tp / (tp + fp + DEFAULT_ATOL),
                    'rec': tp / (tp + fn + DEFAULT_ATOL),
                }
        except Exception as inst:
            print(inst)
    
    return res


def train_and_test_kde(feas_pts, X_test, y_test, n_components=0.95):
    pca = PCA(0.95)
    X_train = pca.fit_transform(feas_pts)
    X_test = pca.transform(X_test)

    params = {
        'bandwidth': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    }
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(X_train)
    kde = grid.best_estimator_
    kde.thresh = np.min(np.exp(kde.score_samples(X_train)))
    y_pred = np.exp(kde.score_samples(X_test))

    y_pred[y_pred > kde.thresh] = 1
    y_pred[y_pred < kde.thresh] = 0
    y_pred[y_pred == kde.thresh] = 1

    acc = 1 - np.mean(np.abs(y_test - y_pred))
    
    print('KDE Test set score = {}'.format(acc))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('TP = {} FP = {}'.format(tp, fp))
    print('FN = {} TN = {}'.format(fn, tn))

    res = {
        'bw': kde.bandwidth,
        'thresh': kde.thresh,
        'acc': acc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'prec': tp / (tp + fp + DEFAULT_ATOL),
        'rec': tp / (tp + fn + DEFAULT_ATOL),
    }
    return res


def sweep_n_tests(fname='', rname=''):
    if fname == '':
        fname = '../miplib/gen-ip021_R1.pickle'
    if rname == '':
        rname = '../results/gen-ip021_R1_{}.csv'.format(np.random.rand())

    summaries = []
    test_size = 1.0
    n_samples = 4000
    n_tests = 10
    MAX_TRIES = 50
    for trial in range(MAX_TRIES):
        if len(summaries) >= n_tests:
            break
        try:
            print('\n'+('*'*80))
            print('Trial {} ({})'.format(trial, len(summaries)))
            print('*'*80)
            res = experiment_MCMC(fname, n_samples=n_samples, test_size=test_size, multiclass=False)
            # res = experiment_MCMC(fname, n_samples=n_samples, test_size=test_size, multiclass=True)
            summaries.append(res)
        except Exception as inst:
            print(inst)
    print('\n\n'+('*'*80))
    print('Summary.')
    print('*'*80)
    print('\n')
    # accuracy
    proj_acc = np.array([ x['proj_res']['acc'] for x in summaries ])
    mcmc_acc = np.array([ x['mcmc_res']['acc'] for x in summaries ])
    gaus_acc = np.array([ x['gaus_res']['acc'] for x in summaries ])
    # recall 
    proj_rec = np.array([ x['proj_res']['rec'] for x in summaries ])
    mcmc_rec = np.array([ x['mcmc_res']['rec'] for x in summaries ])
    gaus_rec = np.array([ x['gaus_res']['rec'] for x in summaries ])
    # precision 
    proj_prec = np.array([ x['proj_res']['prec'] for x in summaries ])
    mcmc_prec = np.array([ x['mcmc_res']['prec'] for x in summaries ])
    gaus_prec = np.array([ x['gaus_res']['prec'] for x in summaries ])
    print('proj:')
    print(proj_acc)
    print('mcmc:')
    print(mcmc_acc)
    print('gaus:')
    print(gaus_acc)
    print('*** proj ***')
    print('Max proj acc: {}'.format(np.max(proj_acc)))
    print('Mean proj acc: {}'.format(np.mean(proj_acc)))
    print('Min proj acc: {}'.format(np.min(proj_acc)))
    print('proj rec: {}'.format(np.mean(proj_rec)))
    print('proj prec: {}'.format(np.mean(proj_prec)))
    print('*** mcmc ***')
    print('Max mcmc acc: {}'.format(np.max(mcmc_acc)))
    print('Mean mcmc acc: {}'.format(np.mean(mcmc_acc)))
    print('Min mcmc acc: {}'.format(np.min(mcmc_acc)))
    print('mcmc rec: {}'.format(np.mean(mcmc_rec)))
    print('mcmc prec: {}'.format(np.mean(mcmc_prec)))
    print('*** gaus ***')
    print('Max gaus acc: {}'.format(np.max(gaus_acc)))
    print('Mean gaus acc: {}'.format(np.mean(gaus_acc)))
    print('Min gaus acc: {}'.format(np.min(gaus_acc)))
    print('gaus rec: {}'.format(np.mean(gaus_rec)))
    print('gaus prec: {}'.format(np.mean(gaus_prec)))
    df = pd.DataFrame(summaries)
    df.to_csv(rname, index=False)
    # exit()


def sweep_test_size():
    fname = 'miplib/gen-ip021_R1.pickle'
    # fname = 'miplib/cod105_R1.pickle'
    # fname = 'miplib/exp-1-500-5-5_R1.pickle'
    # test_sizes = [ 0.98, 0.96, 0.94, 0.92, 0.9, ]
    # test_sizes = [ 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95 ]
    # test_sizes = [ 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.95 ]
    test_sizes = [ 1.0 ]
    # test_sizes = [ 0.1, 0.3, 0.5, 0.7, 0.9 ]
    for test_size in test_sizes:
        print('\n'+('*'*80))
        print(test_size)
        print('*'*80)
        experiment_MCMC(fname, n_samples=n_samples, test_size=test_size, multiclass=False)
    print('DONE.')
    # exit()


def test_toy_probs():
    fname = '../miplib/pentagon.pickle'
    MAX_TRIES=20
    n_tests=10
    n_samples=1000
    test_size=1.0
    summaries = []
    for trial in range(MAX_TRIES):
        if len(summaries) >= n_tests:
            break
        try:
            print('\n'+('*'*80))
            print('Trial {} ({})'.format(trial, len(summaries)))
            print('*'*80)
            res = experiment_MCMC(fname, n_samples=n_samples, test_size=test_size, multiclass=False)
            summaries.append(res)
        except Exception as inst:
            print(inst)
    print('\n\n'+('*'*80))
    print('Summary.')
    print('*'*80)
    print('\n')
    proj_acc = np.array([ x['proj_acc'] for x in summaries ])
    mcmc_acc = np.array([ x['mcmc_acc'] for x in summaries ])
    gaus_acc = np.array([ x['gaus_acc'] for x in summaries ])
    print('proj:')
    print(proj_acc)
    print('mcmc:')
    print(mcmc_acc)
    print('gaus:')
    print(gaus_acc)
    print('')
    print('Max proj acc: {}'.format(np.max(proj_acc)))
    print('Mean proj acc: {}'.format(np.mean(proj_acc)))
    print('Min proj acc: {}'.format(np.min(proj_acc)))
    print('Max mcmc acc: {}'.format(np.max(mcmc_acc)))
    print('Mean mcmc acc: {}'.format(np.mean(mcmc_acc)))
    print('Min mcmc acc: {}'.format(np.min(mcmc_acc)))
    print('Max gaus acc: {}'.format(np.max(gaus_acc)))
    print('Mean gaus acc: {}'.format(np.mean(gaus_acc)))
    print('Min gaus acc: {}'.format(np.min(gaus_acc)))
    
    
    exit()



if __name__ == "__main__":
    # test_toy_probs()
    sweep_n_tests()
    # sweep_test_size()
    exit()
    # sweep_n_samples()
    # fname = 'miplib/gen-ip002_R1.pickle'
    # experiment_MCMC(fname, n_samples=100, test_size=0.5, randomizer=np.random.rand)
    p = Path('../miplib/')
    r = Path('../results/')
    files_to_ignore = [
        Path('../miplib/neos16_R1.pickle'),
        Path('../miplib/markshare_4_0_R1.pickle'),
        Path('../miplib/timtab1CUTS_R1.pickle'), # cannot start hitnrun
    ]
    for fname in p.iterdir():
        if fname in files_to_ignore:
            continue
        if fname.suffix == '.pickle':
            print('\n'+('*'*80))
            print('*'*80)
            print(fname)
            print('*'*80)
            print('*'*80)
            # try:
            rname = r / (fname.stem + '_{}_binary.csv'.format(np.random.rand()))
            sweep_n_tests(fname, rname)
            # experiment_MCMC(fname, n_samples=4000, test_size=0.9)
            # except:
            #     print('Some error.')
