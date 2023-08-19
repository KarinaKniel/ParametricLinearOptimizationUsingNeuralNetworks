import cvxpy as cp
import numpy as np
from fractions import Fraction
import csv
from tqdm import tqdm

dataset_size = 500

# save in csv file
mozart_file = f"mozart_{dataset_size}_c_set.csv"

with open(mozart_file, "a", newline="") as g:
    writer = csv.writer(g)

    for j in tqdm(range(dataset_size)):

        np.random.seed(j)  # ensures reproducibility at each iteration
        p = np.random.uniform(-15, 15)

        # define and solve the LP
        c = np.array([-9 + p, -8])
        A = np.array([[1, 1], [2, 1], [1, 2]])
        b = np.array([6, 11, 9])
        x = cp.Variable(2)
        objective = cp.Minimize(c.T @ x)
        constraints = [A @ x <= b, x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status == "unbounded":
            status = -1
            tup = (float(p), -2, -2, -2, status)
            writer.writerow(tup)
        elif prob.status == "optimal":
            status = 1
            tup = (float(p), x[0].value, x[1].value, prob.value, status)
            writer.writerow(tup)
        else:
            status = 4
