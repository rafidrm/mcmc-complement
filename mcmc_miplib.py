"""
We conduct MCMC complement sampling experiments in R^n using
random instances from MIPLIB.

Generating feasible points:
    - Too lazy to implement hitandrun, so I just randomly generate
    some vertices and then find convex combinations of them.
"""

import numpy as np
from pathlib import Path
import pickle
from pprint import pprint
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import pudb


def vertex_sampler(A_mat, b_vec, randomizer=np.random.rand, n_samples=200):
    m, n = A_mat.shape
    vertices = []
    pts = []
    for _ in tqdm(range(n_samples)):
        c_vec = randomizer(n)
        sol1 = linprog(c_vec, -A_mat, -b_vec, method='interior-point')
        if sol1.success == True:
            vertices.append(np.array(sol1.x))
        else:
            sol2 = linprog(-c_vec, -A_mat, -b_vec, method='interior-point')
            if sol2.success == True:
                vertices.append(np.array(sol2.x))
    vertices = np.array(vertices)
    if len(vertices) < 2:
        raise Exception('Error! Could not find enough solutions.')
    for _ in range(n_samples):
        combo = np.random.dirichlet(np.ones(n_samples))
        pts.append(np.dot(combo, vertices))
    return pts


def load_and_generate_samples(fname, n_samples=200):
    data = pickle.load(open(fname, 'rb'))
    A_mat, b_vec = data['A_mat'], data['b_vec']
    feas_pts = vertex_sampler(A_mat, b_vec, n_samples=n_samples)

    
def direction_sample_helper(con):
    """ Samples a random point from the unit ball then checks with
    the a vector that con'pt > 0
    """
    wrong_direction = 1
    n = len(con)
    while wrong_direction == 1:
        pt = np.random.rand(n) - 0.5
        pt = pt / np.linalg.norm(pt)
        if np.dot(con, pt) >= 0:
            wrong_direction = 0
    return pt


def direction_sample(A_mat, b_vec, bd_pt):
    """ First identifies the relevant constraint on the boundary,
    then calls sample helper.
    """
    ind = list(np.isclose(np.dot(A_mat, bd_pt), b_vec)).index(True)
    con = A_mat[ind]
    return direction_sample_helper(con)


def get_next_bd_pt(A_mat, b_vec, bd_pt, dir_pt):
    """ First removes boundary constraints, then finds nearest
    boundary.
    """
    weights = np.array([(bi - np.dot(ai, bd_pt)) / np.dot(ai, dir_pt) for ai, bi in zip(A_mat, b_vec)])
    weights[weights <= 0] = 99
    weight = min(weights)

    return bd_pt + weight * dir_pt


def shake_n_bake(A_mat, b_vec, init_pt, n_samples=10, scale=2):
    """
    1. randomly sample direction vector (r)
    2. randomly sample magnitude (xi)
    3. add infeasible point (y - xi * r, y)
    4. get next boundary point
    """
    dataset = []
    bd_pt = init_pt
    for ix in tqdm(range(n_samples)): 
        r = direction_sample(A_mat, b_vec, bd_pt)
        xi = np.random.exponential(scale=scale)
        infeas_pt = bd_pt - xi * r
        dataset.append((infeas_pt, bd_pt))
        bd_pt = get_next_bd_pt(A_mat, b_vec, bd_pt, r)
    return dataset


def shake_n_bake_init(A_mat, b_vec):
    m, n = A_mat.shape
    vertices = []
    pts = []
    randomizers = [
        np.random.rand,
        np.random.randn
    ]
    for randomizer in randomizers:
        c_vec = randomizer(n)
        sol1 = linprog(c_vec, -A_mat, -b_vec, method='interior-point')
        if sol1.success == True:
            if np.isclose(np.dot(A_mat, sol1.x), b_vec).any():
                return sol1.x
        sol2 = linprog(-c_vec, -A_mat, -b_vec, method='interior-point')
        if sol2.success == True:
            if np.isclose(np.dot(A_mat, sol2.x), b_vec).any():
                return sol2.x
    raise Exception('Could not initialize shake_n_bake')


def is_feasible(A_mat, b_vec, pt):
    # Assuming inequality form
    tol = 1e-5 * np.ones(len(b_vec))
    if (np.dot(A_mat, pt) >= b_vec - tol).all():
        return True
    else:
        return False


def is_bd(A_mat, b_vec, pt):
    # Assuming inequality form
    tol = 1e-5 * np.ones(len(b_vec))
    if np.isclose(np.dot(A_mat, pt), b_vec).any():
        return True
    else:
        return False


def constraint_projection_sampler(A_mat, b_vec, feas_pts, n_samples=200, scale=2):
    m, n = A_mat.shape
    infeas_pts = []
    for ix in tqdm(range(n_samples)):
        idx = np.random.randint(0, len(feas_pts))
        pt = feas_pts[idx]
        row_idx = np.random.randint(0, m)
        ai, bi = A_mat[row_idx], b_vec[row_idx]
        xi_lb = ( bi - np.dot(ai, pt) ) / ( np.linalg.norm(ai) ** 2 )
        xi = xi_lb - np.random.exponential(scale=scale)
        infeas_pts.append(pt + xi * ai)
    return infeas_pts


def experiment_MCMC(
        fname, 
        n_samples=200,
        randomizer=np.random.rand,
        test_size=0.5,
        scale=2,
    ):
    data = pickle.load(open(fname, 'rb'))
    A_mat, b_vec, b_relax = data['A_mat'], data['b_vec'], data['b_relax']
    print(A_mat.shape)
    feas_pts = vertex_sampler(
        A_mat,
        b_vec,
        n_samples=n_samples,
        randomizer=randomizer
    )
    feas_pts = np.array(feas_pts)
    feas_check = [ is_feasible(A_mat, b_vec, pt) for pt in feas_pts ]
    assert all(feas_check), 'Generated infeasible point instead of feasible'
    train_feas_pts, test_feas_pts, _, _ = train_test_split(
        feas_pts,
        np.ones(len(feas_pts)),
        test_size=test_size
    )

    print('Running projection sampler ...')
    proj_infeas_pts = constraint_projection_sampler(
        A_mat,
        b_relax,
        feas_pts,
        n_samples=n_samples,
        scale=scale,
    )
    feas_check = [ is_feasible(A_mat, b_vec, pt) for pt in proj_infeas_pts ]
    assert all([check is False for check in feas_check]), 'Generated feasible point instead of infeasible'
    train_and_test_classifier(
        train_feas_pts,
        test_feas_pts,
        proj_infeas_pts,
        model=SVC,
    )

    print('Running SB sampler ...')
    init_pt = shake_n_bake_init(A_mat, b_relax)
    dataset = shake_n_bake(
        A_mat,
        b_relax,
        init_pt,
        n_samples=n_samples,
        scale=scale
    )
    mcmc_infeas_pts, bd_pts = zip(*dataset)
    feas_check = [ is_feasible(A_mat, b_vec, pt) for pt in mcmc_infeas_pts ]
    assert all([check is False for check in feas_check]), 'Generated feasible point instead of infeasible'
    train_and_test_classifier(
        train_feas_pts,
        test_feas_pts,
        mcmc_infeas_pts,
        model=SVC,
    )


def train_and_test_classifier(
        train_feas_pts,
        test_feas_pts,
        infeas_pts,
        model=SVC,
    ):
    X = np.vstack((train_feas_pts, infeas_pts))
    train_feas_labels = np.array([np.ones(len(train_feas_pts))]).T
    train_infeas_labels = np.array([np.zeros(len(infeas_pts))]).T
    y = np.vstack(
        (train_feas_labels, train_infeas_labels)
    )
    clf = model()
    clf.fit(X, y)
    # print(clf.predict(test_feas))
    acc = clf.score(test_feas_pts, np.ones(len(test_feas_pts)))
    print('Test set score = {}'.format(acc))



def sweep_test_size():
    fname = 'miplib/cod105_R1.pickle'
    # fname = 'miplib/exp-1-500-5-5_R1.pickle'
    test_sizes = [ 0.98, 0.96, 0.94, 0.92, 0.9, ]
    # test_sizes = [ 0.1, 0.3, 0.5, 0.7, 0.9 ]
    for test_size in test_sizes:
        print('\n'+('*'*80))
        print(test_size)
        print('*'*80)
        experiment_MCMC(fname, n_samples=50, test_size=test_size)
    print('DONE.')
    exit()


if __name__ == "__main__":
    sweep_test_size()
    # fname = 'miplib/gen-ip002_R1.pickle'
    # experiment_MCMC(fname, n_samples=100, test_size=0.5, randomizer=np.random.rand)
    p = Path('miplib/')
    files_to_ignore = [
        Path('miplib/neos16_R1.pickle'),
    ]
    for fname in p.iterdir():
        if fname in files_to_ignore:
            continue
        if fname.suffix == '.pickle':
            print('\n'+('*'*80))
            print(fname)
            print('*'*80)
            experiment_MCMC(fname, n_samples=20, test_size=0.90)
