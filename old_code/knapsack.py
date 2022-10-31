import numpy as np
import random

def knapsack(w, v, max_weight):
    # This initializes our table with the base cases to 0.
    w = [None] + w  # Make the first element None
    v = [None] + v  # Make the first element None
    k = np.array([[0] * (max_weight + 1)] * (len(w)))

    # Loop though all elements of the table, ignoring the base cases
    for i in range(1, len(w)):  # each row
        for j in range(1, max_weight + 1):  # each column
            if w[i] <= j:
                k[i, j] = max(v[i] + k[i - 1, j - w[i]], k[i - 1, j])
            else:
                k[i, j] = k[i - 1, j]
    # Return the table. As the table will be used to reconstruct the solution.
    # The optimal value can be found at the element k[len(w), max_weight]
    return k


def recover_solution(k, w):
    w = [None] + w  # Make the first element None
    i = k.shape[0] - 1
    j = k.shape[1] - 1

    solution = []
    while i > 0 and j > 0:
        # Does not adding item i give us the solution?
        if k[i, j] == k[i - 1, j]:
            i -= 1
        else:
            # Or does adding item i give us the solution
            # In this case we want to the the corresponding index to solution
            solution.append(i-1)
            j -= w[i]
            i -= 1
    # FLip solution because we added things backwards
    return solution[::-1]


def knapsack_solution(w, v, max_weight):
    return recover_solution(knapsack(w, v, max_weight), w)


# w_arr = w = [10, 11, 11, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 11, 11, 11, 11, 11, 11]
# v_arr = [round(x*100) for x in [0.0482715911342968, 0.05748396752689755, 0.039318600122460357, 0.053733799615732436, 0.04745573770621381, 0.04053611498306464, 0.04042265387802357, 0.039592566533574014, 0.039435357845155394, 0.03924694113454463, 0.0374846819544804, 0.03617994633183119, 0.03500427899313985, 0.03500254844656181, 0.03500254844656181, 0.03457932665700525, 0.032742939830836426, 0.03216836713938365, 0.03169001562194884, 0.03169001562194884, 0.03088593088801117, 0.027907935379156812, 0.02677453527294016]]
#
#
# maximum_weight = 40
# ks_sol = knapsack_solution(w_arr, v_arr, maximum_weight)
# ks_opt_val = sum([v_arr[i] for i in ks_sol])
#
#
# print(f"Knapsack DP solution: {ks_sol}")
# print(f"Total value: {ks_opt_val}\n")
# print(w_arr)