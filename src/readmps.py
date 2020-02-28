import numpy as np
from pathlib import Path
import pickle
from pprint import pprint
import pulp
from scipy.optimize import linprog
import pudb 


def OLD_check_lpsolver(A_mat, b_vec):
    m, n = A_mat.shape
    c_vec = np.ones(n)
    # c_vec = np.zeros(n)
    # options = {'presolve': False}
    sol = linprog(c_vec, -A_mat, -b_vec, method='interior-point')
    # sol = linprog(c_vec, -A_mat, -b_vec, options=options)
    if sol.success is True:
        return True
    else:
        return False


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
        # x_vec = np.array([ xi.varValue for xi in x ])
        #return x_vec
    else:
        return False
        #return None


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
                        raise Exception('Error! Column {} split incorrectly'.format(parsed[0]))

            if position == 'RHS':
                if len(parsed) == 3:
                    row = parsed[1]
                    row_dict[row]['rhs'] = float(parsed[2])
                elif len(parsed) == 5:
                    row1, row2 = parsed[1], parsed[3]
                    row_dict[row1]['rhs'] = float(parsed[2])
                    row_dict[row2]['rhs'] = float(parsed[4])
                else:
                    raise Exception('Error! RHS {} split incorrectly'.format(parsed[1]))

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
                    raise Exception('Error! Did not recognize bound {}'.format(parsed[0]))

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


def find_and_replace_mps(pname, make_relaxation=False, n_relaxations=1):
    p = Path(pname)
    for fname in p.iterdir():
        if fname.suffix == '.mps':
            print(fname)
            c_vec, A_mat, b_vec = readmps(fname)
            for ix in range(len(A_mat)):
                tmp_norm = np.linalg.norm(A_mat[ix])
                A_mat[ix] = A_mat[ix] / tmp_norm 
                b_vec[ix] = b_vec[ix] / tmp_norm 
            scale = max(np.max(np.abs(A_mat)), np.max(np.abs(b_vec)))
            print('scale = {}'.format(scale))
            b_vec = b_vec - np.random.exponential(scale=scale, size=len(b_vec))
            print('... read')
            feas_exists = check_lpsolver(c_vec, A_mat, b_vec)
            if feas_exists is True:
                m, n = A_mat.shape
                if make_relaxation is True:
                    for ix in range(n_relaxations):
                        b_relax = b_vec - np.random.exponential(scale=scale, size=m)
                        feas_set = { 
                            'A_mat': A_mat,
                            'b_vec': b_vec,
                            'b_relax': b_relax,
                            'c_vec': c_vec
                        }
                        fname_pickle = p / (fname.stem + '_R{}.pickle'.format(ix+1))
                        with open(fname_pickle, 'wb') as f:
                            pickle.dump(feas_set, f)
                else:
                    feas_set = { 'A_mat': A_mat, 'b_vec': b_vec }
                    fname_pickle = p / (fname.stem + '.pickle')
                    with open(fname_pickle, 'wb') as f:
                        pickle.dump(feas_set, f)


            else:
                print('... is invalid')


def generate_pentagon():
    m = 5
    n = 2
    c_vec = np.ones(2)


if __name__ == "__main__":
    # fname = 'miplib/gen-ip054.mps'
    # readmps(fname)
    pname = '../miplib/'
    find_and_replace_mps(pname, make_relaxation=True)
