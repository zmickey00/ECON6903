import pandas as pd
import numpy as np
from numpy.linalg import solve

# -------------------------------
# Step 0. Import Data Files
# -------------------------------
# Adjust file paths if your files are in a subfolder (e.g., 'data/')
alpha         = pd.read_csv('data/alpha.csv')         # Expected columns: 'Country', 'Sector', 'alpha'
countries_df  = pd.read_csv('data/CountryNames.csv')    # Expected columns: 'Country' (or similar)
deficits      = pd.read_csv('data/Deficits.csv')        # Expected columns: 'Country', 'Deficit'
gamma_io      = pd.read_csv('data/gamma_IO.csv')        # Expected columns: 'Country', 'FromSector', 'ToSector', 'gamma'
gamma_va      = pd.read_csv('data/gamma_VA.csv')        # Expected columns: 'Country', 'Sector', 'gamma_VA'
one_plus_tau  = pd.read_csv('data/one_plus_tau.csv')    # Expected columns: 'Origin', 'Destination', 'Sector', 'one_plus_tau'
pi_df         = pd.read_csv('data/pi.csv')              # Expected columns: 'Origin', 'Destination', 'Sector', 'pi'
va_world      = pd.read_csv('data/VA_World.csv')

J = 40
N = 31

"""read data"""


def obe_plus_tao(k, n, l):
    result =one_plus_tau[(one_plus_tau['sector'] == k & one_plus_tau['CountryDestination'] == n & one_plus_tau['CountryOrigin'] == l)]
    return result


def read_alpha(j, n):
    # return data_alpha
    pass


def read_pi(k, n, l):
    # return data_pi
    pass


def read_gama_1(j, k, n):
    # return data_gama_1
    pass


def read_gama_2(k, n):
    # return data_gama_2
    pass


def read_d(n):
    # return data_d
    pass


def sum_last(j, n, k):
    total = 0
    # index is the l in the format
    for index in range(1, N):
        total += read_alpha(j, n) * read_tao(k, n, index) * read_pi(k, n, index) / (1 + read_tao(k, n, index))

    return total


def big_i(a, b):
    if a == b:
        return 1

    return 0


def item_3(n, i, j, k):
    return big_i(n, i) * sum_last(j, n, k)


def item_2(n, i, j, k):
    pass


def item_1(n, i, j):
    pass


# given n, i, j, k
# total_item = item_1() - item_2 - item_3

# M is a zeros matrix

# define M is 4-D matrix

M = []


def fill_m():
    for j in range(1, J):
        for n in range(1, N):
            for k in range(1, J):
                for i in range(1, N):
                    m_row = (j - 1) * N + n
                    m_col = (k - 1) * N + i
                    m_cell = item_1(n, i, j) - item_2(n, i, j, k) - item_3(n, i, j, k)
                    M[m_row][m_col] = m_cell
    return M


def fill_b(N, J):
    pass


# claim M = zeros
# step 1:
# 1. fill_m
# 2. fill_b
m = fill_m(N, J)

# step 2: get inversed M
inv_m = m ^ -1