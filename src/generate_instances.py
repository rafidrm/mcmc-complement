import numpy as np
from pathlib import Path
import pickle
from pprint import pprint
import pulp
from scipy.optimize import linprog
import sys
from tqdm import tqdm

import pudb 


def check_lpsolver(c_vec, A_mat, b_vec):
    prob = pulp.LpProblem('prob', pulp.LpMinimize)
    m, n = A_mat.shape
    x = [pulp.LpVariable('x{}'.format(ix)) for ix in range(n)]
    prob += pulp.lpSum([ ci * xi for ci, xi in zip(c_vec, x) ])
    for ix in range(m):
        prob += pulp.lpSum([ ai * xi for ai, xi in zip(A_mat[ix], x) ]) >= b_vec[ix], 'con{}'.format(ix)
    prob.solve(pulp.GLPK(msg=0))
    if prob.status in [1, -2]:
        return True
    else:
        return False


def readmps(fname):
    """
    Parses a mpps file to obtain A, b in inequality form: { x | A x >= b }.
    """
    row_dict = {}
    col_names = []
    with open(fname, 'r') as file:
        line = next(file)
        position = None
        while line:
            line = next(file, None)
            parsed = line.split()
            # determine the position
            if len(parsed) == 1:
                if 'ROWS' in line:
                    position = 'ROWS'
                    continue
                if 'COLUMNS' in line:
                    position = 'COLUMNS'
                    continue
                if 'RHS' in line:
                    position = 'RHS'
                    continue
                if 'BOUNDS' in line:
                    position = 'BOUNDS'
                    continue
                if 'ENDATA' in line:
                    break

            if position == 'ROWS':
                row_dict[parsed[1]] = {
                    'name': parsed[1],
                    'cols': {},
                    'rhs': 0,
                    'sign': parsed[0],
                }

            if position == 'COLUMNS':
                # remove title rows
                if 'MARK' in parsed[0] or 'INT' in parsed[2]:
                    continue
                else:
                    col_names.append(parsed[0])
                    if len(parsed) == 3:
                        row = parsed[1]
                        row_dict[row]['cols'][parsed[0]] = float(parsed[2])
                    elif len(parsed) == 5:
                        row1, row2 = parsed[1], parsed[3]
                        row_dict[row1]['cols'][parsed[0]] = float(parsed[2])
                        row_dict[row2]['cols'][parsed[0]] = float(parsed[4])
                    else:
                        raise Exception('Column {} split incorrectly'.format(parsed[0]))

            if position == 'RHS':
                if len(parsed) == 3:
                    row = parsed[1]
                    row_dict[row]['rhs'] = float(parsed[2])
                elif len(parsed) == 5:
                    row1, row2 = parsed[1], parsed[3]
                    row_dict[row1]['rhs'] = float(parsed[2])
                    row_dict[row2]['rhs'] = float(parsed[4])
                else:
                    raise Exception('RHS {} split incorrectly'.format(parsed[1]))

            if position == 'BOUNDS':
                col_names.append(parsed[2])
                if parsed[0] in ['LI', 'LB', 'LO']:
                    row_dict[parsed[0] + parsed[2]] = {
                        'name': parsed[0] + parsed[2],
                        'cols': {parsed[2]: 1},
                        'rhs': float(parsed[3]),
                        'sign': 'G',
                    }
                elif parsed[0] in ['UI', 'UB', 'UP']:
                    row_dict[parsed[0] + parsed[2]] = {
                        'name': parsed[0] + parsed[2],
                        'cols': {parsed[2]: -1},
                        'rhs': -1 * float(parsed[3]),
                        'sign': 'G',
                    }
                elif parsed[0] in ['EQ', 'FX']:
                    row_dict[parsed[0] + parsed[2] + 'UB'] = {
                        'name': parsed[0] + parsed[2] + 'UB',
                        'cols': {parsed[2]: 1},
                        'rhs': 1 * float(parsed[3]),
                        'sign': 'G',
                    }
                    row_dict[parsed[0] + parsed[2] + 'LB'] = {
                        'name': parsed[0] + parsed[2] + 'LB',
                        'cols': {parsed[2]: -1},
                        'rhs': -1 * float(parsed[3]),
                        'sign': 'G',
                    }
                elif parsed[0] in ['PL']:
                    row_dict[parsed[0] + parsed[2]] = {
                        'name': parsed[0] + parsed[2],
                        'cols': {parsed[2]: 1},
                        'rhs': 0,
                        'sign': 'G',
                    }
                elif parsed[0] in ['FR']:
                    continue
                else:
                    raise Exception('Did not recognize bound {}'.format(parsed[0]))

    col_names = sorted(list(set(col_names)))
    rows_to_remove = []
    c_dict = {}
    new_rows = [] 
    for row in row_dict:
        # create a new entry for column vector
        row_dict[row]['row_vec'] = { col: 0 for col in col_names }
        for col in row_dict[row]['cols']:
            row_dict[row]['row_vec'][col] = row_dict[row]['cols'][col]
        if row_dict[row]['sign'] == 'N':
            row_dict[row]['row_vec'][col] = row_dict[row]['cols'][col]
            c_dict = row_dict[row].copy()
            rows_to_remove.append(row)
        # if LEQ constraint, then swap signs
        if row_dict[row]['sign'] == 'L':
            for col in row_dict[row]['row_vec']:
                row_dict[row]['row_vec'][col] *= -1
            row_dict[row]['rhs'] *= -1 
            row_dict[row]['sign'] = 'G'
        # if equality constraint, then create two new versions
        if row_dict[row]['sign'] == 'E':
            rows_to_remove.append(row)
            ub_row = row_dict[row].copy()
            ub_row['sign'] = 'L'
            new_rows.append((row + 'UB', ub_row))
            lb_row = row_dict[row].copy()
            lb_row['sign'] = 'G'
            new_rows.append((row + 'LB', lb_row))

    for row in rows_to_remove:
        row_dict.pop(row)
    for key, val in new_rows:
        row_dict[key] = val

    row_names = sorted(list(row_dict.keys()))
    m, n = len(row_names), len(col_names)
    c_vec = np.zeros(n)
    A_mat = np.zeros((m, n))
    b_vec = np.zeros(m)
    for ix in range(m):
        row = row_names[ix]
        for iy in range(n):
            col = col_names[iy]
            A_mat[ix, iy] = row_dict[row]['row_vec'][col]
        b_vec[ix] = row_dict[row]['rhs']
    for ix in range(n):
        c_vec[ix] = c_dict['row_vec'][col_names[iy]]

    return c_vec, A_mat, b_vec


def add_box_bounds(c_vec, A_mat, b_vec):
    A_bounded = list(A_mat.copy())
    b_bounded = list(b_vec.copy())

    prob = pulp.LpProblem('prob', pulp.LpMinimize)
    m, n = A_mat.shape
    x = [pulp.LpVariable('x{}'.format(ix)) for ix in range(n)]
    prob += pulp.lpSum([ ci * xi for ci, xi in zip(c_vec, x) ])
    for ix in range(m):
        prob += pulp.lpSum([ ai * xi for ai, xi in zip(A_mat[ix], x) ]) >= b_vec[ix], 'con{}'.format(ix)
    prob.solve(pulp.GLPK(msg=0))
    if prob.status in [1, -2]:  
        bound = np.max(np.abs([xi.varValue for xi in x])) * 1.1
    else:
        raise Exception('Failed to add box bounds. Problem infeasible.')

    for ix in range(n):
        row_ub = np.zeros(n)
        row_ub[ix] = -1
        A_bounded.append(row_ub)
        b_bounded.append(-1 * bound)
        row_lb = np.zeros(n)
        row_lb[ix] = 1
        A_bounded.append(row_lb)
        b_bounded.append(-1 * bound)

    A_new = np.array(A_bounded)
    b_new = np.array(b_bounded)

    feas_exists = check_lpsolver(c_vec, A_new, b_new)
    assert feas_exists == True, 'Box bounds made the problem infeasible.'
    return A_new, b_new


def safely_remove_rows(c_vec, A_relax, b_relax, n_components=0.8):
    m_relax, n_relax = A_relax.shape
    MAX_TRIES = m_relax
    m_curr = m_relax
    _tries = 0
    A_curr = A_relax.copy()
    b_curr = b_relax.copy()
    m_curr, _ = A_curr.shape
    min_size = max(n_components * m_relax, 2 * n_relax) + 1 
    if n_components > 0.95:
        m_curr = 0
    while m_curr > min_size:
        # only remove rows that represent real constraints
        row_to_rm = np.random.choice(m_curr - 2 * n_relax)

        A_tmp = list(A_curr.copy())
        A_tmp.pop(row_to_rm)
        A_tmp = np.array(A_tmp)
        b_tmp = list(b_curr.copy())
        b_tmp.pop(row_to_rm)
        b_tmp = np.array(b_tmp)

        feas_exists = check_lpsolver(c_vec, A_tmp, b_tmp)
        if feas_exists == True:
            A_curr = A_tmp
            b_curr = b_tmp
            m_curr, _ = A_curr.shape
            sys.stdout.write('.')
        else:
            continue

        # break conditions
        _tries += 1
        if _tries >= MAX_TRIES:
            break

    m, n = A_curr.shape
    print(
        'Final version is {}, {} from {}, {}'.format(m, n, m_relax, n_relax)
    )
    print('Shrunk to {} of original size'.format(m/m_relax))
    return A_curr, b_curr 


def generate_instances_by_cuts(pname, n_instances=1, n_components=0.8):
    p = Path(pname)
    for fname in p.iterdir():
        if fname.suffix == '.mps':
            print('*'*80)
            print(fname)
            print('*'*80)
            c_vec, A_mat, b_vec = readmps(fname)
            # normalize and make sure it is full dimensional
            for ix in range(len(A_mat)):
                tmp_norm = np.linalg.norm(A_mat[ix])
                A_mat[ix] = A_mat[ix] / tmp_norm
                b_vec[ix] = b_vec[ix] / tmp_norm
            scale = max(np.max(np.abs(A_mat)), np.max(np.abs(b_vec)))
            print('scale = {}'.format(scale))
            b_vec = b_vec - scale
            # create relaxation
            check_feas = check_lpsolver(c_vec, A_mat, b_vec)
            if check_feas == False:
                'This problem is already infeasible.'
                continue
            # make sure relaxed versions are bounded and closed
            A_mat, b_vec = add_box_bounds(c_vec, A_mat, b_vec)
            # create cut instances
            for inst in range(n_instances):
                A_relax, b_relax = safely_remove_rows(
                    c_vec,
                    A_mat,
                    b_vec,
                    n_components=n_components
                )
                feas_set = {
                    'A_mat': A_mat,
                    'b_vec': b_vec,
                    'A_relax': A_relax,
                    'b_relax': b_relax,
                    'c_vec': c_vec,
                }
                fname_pickle = p / (
                    fname.stem + '_cut{}_R{}.pickle'.format(n_components, inst + 1)
                )
                with open(fname_pickle, 'wb') as f:
                    pickle.dump(feas_set, f)


def generate_instances_by_perturbations(pname, n_instances=1, scale_factor=1):
    p = Path(pname)
    for fname in p.iterdir():
        if fname.suffix == '.mps':
            print('*'*80)
            print(fname)
            print('*'*80)
            c_vec, A_mat, b_vec = readmps(fname)
            # normalize and make sure it is full dimensional
            for ix in range(len(A_mat)):
                tmp_norm = np.linalg.norm(A_mat[ix])
                A_mat[ix] = A_mat[ix] / tmp_norm
                b_vec[ix] = b_vec[ix] / tmp_norm
            scale = max(np.max(np.abs(A_mat)), np.max(np.abs(b_vec)))
            print('scale = {}'.format(scale))
            b_vec = b_vec - scale
            # create relaxation
            check_feas = check_lpsolver(c_vec, A_mat, b_vec)
            if check_feas == False:
                'This problem is already infeasible.'
                continue
            # make sure relaxed versions are bounded and closed
            A_mat, b_vec = add_box_bounds(c_vec, A_mat, b_vec)
            # create cut instances
            for inst in range(n_instances):
                relax_scale = scale * scale_factor  # just use the same scale as the expansion
                print('relax scale = {}'.format(relax_scale))
                A_relax = A_mat.copy()
                b_relax = b_vec - np.random.exponential(scale=relax_scale, size=len(b_vec))

                feas_set = {
                    'A_mat': A_mat,
                    'b_vec': b_vec,
                    'A_relax': A_relax,
                    'b_relax': b_relax,
                    'c_vec': c_vec,
                }
                fname_pickle = p / (
                    fname.stem + '_scale{}_R{}.pickle'.format(scale_factor, inst + 1)
                )
                with open(fname_pickle, 'wb') as f:
                    pickle.dump(feas_set, f)



if __name__ == "__main__":
    pname = '../miplib3/'
    # generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.05)
    generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.1)
    # generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.15)
    generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.2)
    # generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.25)
    generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.3)
    # generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.35)
    generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.4)
    # generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.45)
    generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.5)
    # generate_instances_by_perturbations(pname, n_instances=1, scale_factor=0.0)
