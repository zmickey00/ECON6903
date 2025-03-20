import pandas as pd
import numpy as np
from numpy.linalg import solve

# -------------------------------
# Step 0. Import Data Files
# -------------------------------
# Adjust file paths if your files are in a subfolder (e.g., 'data/')
alpha         = pd.read_csv('alpha.csv')         # Expected columns: 'country', 'sector', 'alpha'
countries_df  = pd.read_csv('CountryNames.csv')    # Expected columns: 'Country'
deficits      = pd.read_csv('Deficits.csv')        # Expected columns: 'country', 'Deficits'
gamma_io      = pd.read_csv('gamma_IO.csv')        # Expected columns: 'country', 'SectorOrigin', 'SectorDestination', 'gamma_IO'
gamma_va      = pd.read_csv('gamma_VA.csv')        # Expected columns: 'country', 'sector', 'gamma_VA'
one_plus_tau  = pd.read_csv('one_plus_tau.csv')    # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'one_plus_tau'
pi_df         = pd.read_csv('pi.csv')              # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'pi'
va_world      = pd.read_csv('VA_World.csv')
#Normalization
world_va = va_world['VA_World'].iloc[0]   # Assuming a single value is provided
deficits['Deficits'] = deficits['Deficits'] / world_va

# print(va_world.columns)

J = 40  # number of sectors (or whichever grouping you have)
N = 31  # number of countries (or whichever grouping you have)

# -------------------------------------
# Precompute lookup dictionaries for speed
# -------------------------------------

# For alpha, key: (sector, country)
alpha_dict = { (row['sector'], row['country']): row['alpha'] for _, row in alpha.iterrows() }

# For one_plus_tau, key: (sector, destination, origin)
one_plus_tau_dict = { (row['sector'], row['CountryDestination'], row['CountryOrigin']): row['one_plus_tau']
                      for _, row in one_plus_tau.iterrows() }

# For pi, key: (sector, destination, origin)
pi_dict = { (row['sector'], row['CountryDestination'], row['CountryOrigin']): row['pi']
            for _, row in pi_df.iterrows() }

# For gamma_io, key: (SectorOrigin, SectorDestination, country)
gamma_io_dict = { (row['SectorOrigin'], row['SectorDestination'], row['country']): row['gamma_IO']
                  for _, row in gamma_io.iterrows() }

# For gamma_va, key: (sector, country)
gamma_va_dict = { (row['sector'], row['country']): row['gamma_VA']
                  for _, row in gamma_va.iterrows() }

# For deficits, key: country
deficits_dict = { row['country']: row['Deficits'] for _, row in deficits.iterrows() }

def read_alpha(sector, country):
    return alpha_dict[(sector, country)]

def one_plus_tao(sector, destination, origin):
    return one_plus_tau_dict[(sector, destination, origin)]

def read_pi(sector, destination, origin):
    return pi_dict[(sector, destination, origin)]

def read_gama_1(sorigin, sdestination, country):
    return gamma_io_dict[(sorigin, sdestination, country)]

def read_gama_2(sector, country):
    return gamma_va_dict[(sector, country)]

def read_d(country):
    return deficits_dict[country]


M = np.full((J * N + 1, J * N),20.0)

print(M)
def sum_last(j, n, k):
    total = 0
    # index represents the origin country (l)
    for index in range(1, N):  # assumes 1-based indexing for country indices
        total += (read_alpha(j, n) *
                  (one_plus_tao(k, n, index) - 1) *
                  read_pi(k, n, index)) / one_plus_tao(k, n, index)
    return total

def big_i(a, b):
    if a == b:
        return 1
    else:
        return 0

def item_3(n, i, j, k):
    return big_i(n, i) * sum_last(j, n, k)

def item_2(n, i, j, k):
    return (read_pi(k, i, n) / one_plus_tao(k, i, n)) * (read_gama_1(j, k, n) + read_alpha(j, n) * read_gama_2(k, n))

def item_1(n, i, j, k):
    return big_i((j - 1) * N + n, (k - 1) * N + i)

def m_entry(n, i, j, k):
    return item_1(n, i, j, k) - item_2(n, i, j, k) - item_3(n, i, j, k)


def fill_m():
    # Loop over sectors and countries in 0-based indexing.
    for j in range(J):  # j = 0,1,...,J-1; sector = j+1
        for n in range(N):  # n = 0,1,...,N-1; country = n+1
            for k in range(J):  # k = 0,1,...,J-1; sector = k+1
                for i in range(N):  # i = 0,1,...,N-1; country = i+1
                    m_row = j * N + n
                    m_col = k * N + i
                    # Adjust indices to 1-based when calling m_entry and related functions
                    M[m_row, m_col] = m_entry(n + 1, i + 1, j + 1, k + 1)
    return M

#%%
fill_m()


print(M)

# def b_entry(j,n):
#    pass


# print(read_pi(2, 3,  2))
# print(one_plus_tao(2, 3,  2))
# print(read_gama_1(1,2,2))
# print(read_gama_2(2,2))
# print(read_alpha(1,2))
# function b=fillb()
#    for n =1:N
#        for j =1:J
#            b_index = (j - 1) * N + n;
#            b(b_index)=alphaLookupNumeric(j,n)*deficitLookupNumeric(n);
#        end
#    end
# end
# Optionally, print or inspect a portion of M

#
# # Convert the NumPy array M to a DataFrame
# M_df = pd.DataFrame(M_filled)
#
# # Export the DataFrame to a CSV file
# M_df.to_csv('M.csv', index=False)


B = np.zeros((J * N + 1, 1))


def product(j, n):
    return read_alpha(j, n) * read_d(n)


def fill_b():
    for j in range(J):
        for n in range(N):
            b_row = j * N + n
            B[b_row, 0] = product(j + 1, n + 1)

    return B


fill_b()
#
# print(B)
#
# # %%
# print(M[J * N, 3])

# %%
# Replace the row due to walras law
for k in range(J):
    for i in range(N):
        m_col = k * N + i
        for n in range(N):
            M[J * N, m_col] += read_gama_2(k + 1, n + 1) * read_pi(k + 1, i + 1, n + 1) / one_plus_tao(k + 1, i + 1,
                                                                                                       n + 1)

print(M)

# %%
B[J * N, 0] = 1
print(B)

# %%
M_reduced = np.delete(M, 1238, axis=0)
B_reduced = np.delete(B, 1238, axis=0)

print(M_reduced.shape)
print(B_reduced.shape)

# %%
M_inv = np.linalg.inv(M_reduced)
print(M_inv)

# %%
X = np.dot(M_inv, B_reduced)
print(X)
#
# print(X.shape)
#
# # %%
# print(pi_df.shape)
#
# # %%
# sum = 0
# for i in range(40):
#     sum += X[i, 0]

# print(sum)
#
# # %%
# print(X[93, 0])
# print(read_pi(1, 3, 2))
#
# # %%
# print(read_gama_1(1, 2, 3))
#
# # %%
# print(one_plus_tao(20, 12, 25))

# %%
# X_country = np.zeros((31, 1))
# for i in range(31):
#     sum = 0
#     for j in range(40):
#         sum += X[31 * i + j]
#     X_country[i] = sum
#
# print(X_country)