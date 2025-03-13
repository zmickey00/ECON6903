from cgitb import reset

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


def one_plus_tao(sector, destination, origin):
    result =one_plus_tau[(one_plus_tau['sector'] == sector) & (one_plus_tau['CountryDestination'] == destination) & (one_plus_tau['CountryOrigin'] == origin)]['one_plus_tau'].iloc[0]
    return result

def read_alpha(sector, country):
    result = alpha[(alpha['sector'] == sector) & (alpha['country'] == country)]['alpha'].iloc[0]
    return result
def read_pi(sector, destination, origin):
    result = pi_df[(pi_df['sector'] == sector) & (pi_df['CountryDestination'] == destination) & (pi_df['CountryOrigin'] == origin)] ['pi'].iloc[0]
    return result
def read_gama_1(sorigin, sdestination, country):
    result = gamma_io[(gamma_io['SectorOrigin'] == sorigin) & (gamma_io['SectorDestination'] == sdestination) & (gamma_io['country'] == country)] ['gamma_IO'].iloc[0]
    return result
def read_gama_2(sector, country):
    result = gamma_va[(gamma_va['sector'] == sector) & (gamma_va['country'] == country)] ['gamma_VA'].iloc[0]
    return result


def read_d(country):
    result = deficits[(deficits['country'] == country)]['Deficits'].iloc[0]
    return result
# print(one_plus_tao(1,2,3))
# print(read_alpha(6,1))
# print(read_pi(1,1,3))
# print(read_gama_1(2,35,1))
# print(read_gama_2(2,1))
# print(read_d(5))


def sum_last(j, n, k):
    total = 0
    # index is the l in the format
    for index in range(1, N):
        total += (read_alpha(j, n) * (one_plus_tao(k,n,index) - 1) * read_pi(k, n, index)) / (one_plus_tao(k, n, index))
    return total

def big_i(a, b):
    if a == b:
        return 1
    return 0

def item_3(n, i, j, k):
    return big_i(n, i) * sum_last(j, n, k)

def item_2(n, i, j, k):
    return (read_pi(k,i,n)/one_plus_tao(k,i,n)) * (read_gama_1(j,k,n) + read_alpha(j,n) * read_gama_2(k,n))

def item_1(n, i, j, k):
    return big_i( (j-1) * N + n  ,  (k-1) * N + i )

def m_entry(n,i,j,k):
    return item_1(n,i,j,k) - item_2(n,i,j,k) - item_3(n,i,j,k)

M = np.zeros((1240, 1240))

def fill_m():
    for j in range(0, J):
        for n in range(0, N):
            for k in range(0, J):
                for i in range(0, N):
                    m_row = (j - 1) * N + n
                    m_col = (k - 1) * N + i
                    M[m_row][m_col] = m_entry(n+1,i+1,j+1,k+1)
    return M

fill_m()