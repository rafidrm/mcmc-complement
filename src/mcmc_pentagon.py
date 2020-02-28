import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, minimize

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.linewidth'] = 2 
mpl.rcParams['backend'] = 'ps'
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = 8., 8.
mpl.rcParams['font.family'] = 'serif' 
mpl.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
# mpl.rcParams['font.size'] = 36
mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.fontsize'] = 24 
mpl.rcParams['legend.loc'] = 'lower center'
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['lines.linewidth'] = 3 
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.markeredgewidth'] = 3
# mpl.rcParams['lines.markeredgewidth'] = 3.5
mpl.rcParams['markers.fillstyle'] = 'none'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['xtick.labelsize'] = 36
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['xtick.major.width'] = 1.6
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['ytick.labelsize'] = 36
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['ytick.major.width'] = 1.6
mpl.rcParams['ytick.right'] = True 




def a1_hyperplane(x):
    return x[0] * (4 - 1.26) + x[1] * (0 - 3.79)

def a2_hyperplane(x):
    return x[0] * (1.26 + 3.2) + x[1] * (3.79 - 2.39)

def a3_hyperplane(x):
    return x[0] * (-3.2 + 3.28) + x[1] * (2.39 + 2.28)

def a4_hyperplane(x):
    return x[0] * (-3.28 - 1.13) + x[1] * (-2.28 + 3.83)

def a5_hyperplane(x):
    return x[0] * (1.13 - 4) + x[1] * (-3.83 - 0)


def get_pentagon_feasible_set():
    ''' Generates the A matrix and b vector by matrix inversion.
    '''
    X1 = np.array([
        [4, 0],
        [1.26, 3.79],
    ])
    X2 = np.array([
        [1.26, 3.79],
        [-3.2, 2.39],
    ])
    X3 = np.array([
        [-3.2, 2.39],
        [-3.28, -2.28],
    ])
    X4 = np.array([
        [-3.28, -2.28],
        [1.13, -3.83],
    ])
    X5 = np.array([
        [1.13, -3.83],
        [4, 0],
    ])
    Xs = [X1, X2, X3, X4, X5]
    ones = np.array([1, 1])

    A_mat = []
    b_vec = -1 * np.ones(5)
    for X in Xs:
        A_mat.append(np.linalg.solve(X, ones))
    A_mat = np.array(A_mat)
    A_mat = -1 * A_mat
    print(A_mat)
    print(b_vec)
    return A_mat, b_vec


def direction_sample_helper(con):
    ''' Samples a random point from the unit ball then checks with
    the a vector that con'pt > 0
    '''
    wrong_direction = 1
    while wrong_direction == 1:
        pt = np.random.rand(2) - 0.5
        pt = pt / np.linalg.norm(pt)
        if np.dot(con, pt) >= 0:
            wrong_direction = 0
    return pt


def direction_sample(A_mat, bd_pt):
    ''' First identifies the relevant constraint on the boundary,
    then calls sample helper.
    '''
    ind = list(np.isclose(np.dot(A_mat, bd_pt), -1)).index(True)
    con = A_mat[ind]
    return direction_sample_helper(con)


def get_next_bd_pt(A_mat, bd_pt, dir_pt):
    ''' First removes boundary constraints, then finds nearest
    boundary.
    '''
    weights = np.array([(-1 - np.dot(ai, bd_pt)) / np.dot(ai, dir_pt) for ai in A_mat])
    weights[weights <= 0] = 99
    weight = min(weights)

    return bd_pt + weight * dir_pt


def shake_n_bake(A_mat, init_pt, n=10):
    '''
    1. randomly sample direction vector (r)
    2. randomly sample magnitude (xi)
    3. add infeasible point (y - xi * r, y)
    4. get next boundary point
    '''
    dataset = []
    bd_pt = init_pt
    while len(dataset) < n:
        r = direction_sample(A_mat, bd_pt)
        xi = np.random.exponential(scale=2)
        infeas_pt = bd_pt - xi * r
        dataset.append((infeas_pt, bd_pt))
        bd_pt = get_next_bd_pt(A_mat, bd_pt, r)
    return dataset


def experiment_pentagon(save_dir='', n_samples=500):
    ''' Run the experiment.
    '''
    A_mat, b_vec = get_pentagon_feasible_set()
    init_pt = 0.5 * np.array([4, 0]) + 0.5 * np.array([1.26, 3.79])
    dataset = shake_n_bake(A_mat, init_pt, n=n_samples)
    infeas_pts, bd_pts = zip(*dataset)
    x, y = zip(*infeas_pts)
    xs = [
        # x[:5],
        x[:50],
        x[:500],
    ]
    ys = [
        # y[:5],
        y[:50],
        y[:500],
    ]


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))


    for i, ax in enumerate(axs):
        # plot the polyhedron lines
        xpoly = [4, 1.26, -3.2, -3.28, 1.13, 4]
        ypoly = [0, 3.79, 2.39, -2.28, -3.83, 0]
        ax.plot(
            xpoly, ypoly, markersize=0, color='black', 
        )

        # then plot sampled points
        ax.scatter(
            xs[i], ys[i],
        )

        ax.tick_params(direction='in')
        ax.grid(True, ls='--', alpha=0.1)
        ax.set(
            xlabel='x', 
            ylabel='y', 
            xlim=(-10, 10),
            xticks=[-10, -5, 0, 5, 10],
            ylim=(-10, 10),
            yticks=[-10, -5, 0, 5, 10],
        )
    save_settings = {
        'bbox_inches': 'tight',
        'dpi': 300,
    }
    fig.savefig(save_dir + 'pentA.png', **save_settings)
    fig.savefig(save_dir + 'pentA.eps', **save_settings)


if __name__ == "__main__":
    save_dir = 'img/'
    experiment_pentagon(save_dir)

