from scipy.optimize import linear_sum_assignment
import numpy as np


# ------------------------------------------------------------------------------------------------

def match(scores, threshold=0):
    """
    Performs matching of elements in two sets, A, B.
    The match scores should be a matrix with
    scores[i, j] = match score between A_i and B_j.

    Each element in A is assigned to at most one element
    in B and vice-versa, so as to maximize the sum of
    matching scores. Matches with score below a threshold
    are not considered.

    The output consists of two lists, a list of elements from
    the set A, and a list (with the same size) of elements from
    the set B, with elements on the same position being matched.
    """

    if type(scores) is not np.ndarray:
        costs = -np.array(scores)
    else:
        costs = -scores

    # linear_sum_assignment minimizes the cost, hence the minus sign
    row_ind, col_ind = linear_sum_assignment(costs)
    n = len(row_ind)

    # TODO: this post-matching filter might not be optimal
    # filter matches with score below a specified threshold
    out_row = []
    out_col = []
    for i in range(n):
        if scores[row_ind[i]][col_ind[i]] > threshold:
            out_row.append(row_ind[i])
            out_col.append(col_ind[i])

    return out_row, out_col


# ------------------------------------------------------------------------------------------------

def _test():
    cost = np.array([[4, 1, 3, 2], [2, 0, 5, 1], [3, 2, 2, 1], [1, 2, 3, 1]])
    row_ind, col_ind = match(cost)
    print('row indices: {}'.format(row_ind))
    print('column indices: {}'.format(col_ind))
    print('maximal cost sum: {}'.format(cost[row_ind, col_ind].sum()))


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    _test()

# ------------------------------------------------------------------------------------------------
