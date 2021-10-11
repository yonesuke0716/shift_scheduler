import numpy as np
import itertools

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

import pulp

problem = pulp.LpProblem('test', pulp.LpMaximize)
a = pulp.LpVariable('a', 0, 1)
b = pulp.LpVariable('b', 0, 1)
problem += a + b

status = problem.solve()
print(a.value())
print(b.value())