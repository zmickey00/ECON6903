import pandas as pd
import numpy as np
from numpy.linalg import solve

# -------------------------------
# Step 0. Import Data Files
# -------------------------------
alpha_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\alpha.csv')
countries_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\CountryNames.csv')
deficits_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\Deficits.csv')
gamma_io_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\gamma_IO.csv')
gamma_va_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\gamma_VA.csv')
one_plus_tau_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\one_plus_tau.csv')
pi_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\pi.csv')
va_world_df = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\VA_World.csv')

# -------------------------------
# Step 1. Data Preprocessing
# -------------------------------
world_va = va_world_df['VA_World'].iloc[0]
print("World Value Added =", world_va)

# Normalize deficits by world value added.
deficits_df['Deficits'] = deficits_df['Deficits'] / world_va

# Build lists of countries and sectors.
countries = countries_df['Country'].tolist()
sectors = sorted(alpha_df['sector'].unique().tolist())

N = len(countries)   # Number of countries
J = len(sectors)     # Number of sectors
size = N * J         # Total number of unknowns

# Helper: map (country index, sector index) to vector index.
def idx(n, j):
    return n * J + j

# -------------------------------
# Step 2. Precompute Dictionaries for Fast Lookups
# -------------------------------
alpha_dict = { (row['country'], row['sector']): row['alpha']
               for _, row in alpha_df.iterrows() }

deficit_dict = { row['country']: row['Deficits']
                 for _, row in deficits_df.iterrows() }

gamma_io_dict = { (row['country'], row['SectorOrigin'], row['SectorDestination']): row['gamma_IO']
                  for _, row in gamma_io_df.iterrows() }

gamma_va_dict = { (row['country'], row['sector']): row['gamma_VA']
                  for _, row in gamma_va_df.iterrows() }

one_plus_tau_dict = { (row['CountryOrigin'], row['CountryDestination'], row['sector']): row['one_plus_tau']
                      for _, row in one_plus_tau_df.iterrows() }

pi_dict = { (row['CountryOrigin'], row['CountryDestination'], row['sector']): row['pi']
            for _, row in pi_df.iterrows() }

pi_tariff_dict = { (row['CountryOrigin'], row['CountryDestination'], row['sector']): row['pi']
                   for _, row in pi_df.iterrows() if 'CountryOrigin' in row }

# -------------------------------
# Step 3. Construct Market Clearing Equations (A_market, b_market)
# -------------------------------
A_market = np.zeros((size, size))
b_market = np.zeros(size)

for n, country in enumerate(countries):
    for j, sector_j in enumerate(sectors):
        eq_idx = idx(n, j)
        # Identity term.
        A_market[eq_idx, eq_idx] = 1.0

        # (i) Trade contributions via gamma_IO.
        for k, sector_k in enumerate(sectors):
            key_gamma = (country, sector_j, sector_k)
            if key_gamma not in gamma_io_dict:
                continue
            gamma_val = gamma_io_dict[key_gamma]
            for i, country_i in enumerate(countries):
                key_tau = (country_i, country, sector_k)
                key_pi = (country_i, country, sector_k)
                if key_tau in one_plus_tau_dict and key_pi in pi_dict:
                    tau_val = one_plus_tau_dict[key_tau]
                    pi_val = pi_dict[key_pi]
                    coeff = gamma_val * (pi_val / tau_val)
                    A_market[eq_idx, idx(i, k)] -= coeff

        # (ii) Preference and labor share contributions.
        alpha_val = alpha_dict.get((country, sector_j), 0.0)
        # (a) Contribution from gamma_VA.
        for j_prime, sector_jprime in enumerate(sectors):
            gamma_va_val = gamma_va_dict.get((country, sector_jprime), 0.0)
            for i, country_i in enumerate(countries):
                key_tau = (country, country_i, sector_jprime)
                key_pi = (country, country_i, sector_jprime)
                if key_tau in one_plus_tau_dict and key_pi in pi_dict:
                    tau_val = one_plus_tau_dict[key_tau]
                    pi_val = pi_dict[key_pi]
                    coeff = alpha_val * gamma_va_val * (pi_val / tau_val)
                    A_market[eq_idx, idx(i, j_prime)] -= coeff

        # (b) Tariff revenue contribution.
        for j_prime, sector_jprime in enumerate(sectors):
            inner_sum = 0.0
            for i, country_i in enumerate(countries):
                key_tau = (country, country_i, sector_jprime)
                key_pi = (country, country_i, sector_jprime)
                if key_tau in one_plus_tau_dict and key_pi in pi_tariff_dict:
                    base_tau = one_plus_tau_dict[key_tau]
                    tau_val = base_tau - 1.0
                    pi_val = pi_tariff_dict[key_pi]
                    inner_sum += (tau_val * pi_val) / base_tau
            coeff = alpha_val * inner_sum
            A_market[eq_idx, idx(n, j_prime)] -= coeff

        # (iii) Right-hand side from deficits.
        deficit_val = deficit_dict.get(country, 0.0)
        b_market[eq_idx] = alpha_val * deficit_val

# -------------------------------
# Step 4. Construct the Normalization Equation
# -------------------------------
# Normalization: ∑_(n,j) gamma_VA(country, sector) * X^n_j = 1.
A_norm = np.zeros((1, size))
for n, country in enumerate(countries):
    for j, sector in enumerate(sectors):
        gamma_va_val = gamma_va_dict.get((country, sector), 0.0)
        A_norm[0, idx(n, j)] = gamma_va_val
b_norm = np.array([1.0])

# -------------------------------
# Step 5. Form Full System and Drop a Redundant Equation
# -------------------------------
# Stack the market clearing equations and normalization row.
A_full = np.vstack([A_market, A_norm])
b_full = np.concatenate([b_market, b_norm])

print("A_full shape:", A_full.shape)  # Should be (size+1, size)
print("b_full shape:", b_full.shape)  # Should be (size+1,)

rank_reduced1 = np.linalg.matrix_rank(A_full)
print("Rank of reduced system:", rank_reduced1, "vs. size:", A_full.shape[0])

# Due to Walras' law, one market clearing equation is redundant.
# Drop one market clearing row. Here, we drop row index 0.
drop_row = 0
A_reduced = np.delete(A_full, drop_row, axis=0)
b_reduced = np.delete(b_full, drop_row, axis=0)

print("A_reduced shape:", A_reduced.shape)  # Should be (size, size)
print("b_reduced shape:", b_reduced.shape)

# Check rank of the reduced system.
rank_reduced2 = np.linalg.matrix_rank(A_reduced)
print("Rank of reduced system:", rank_reduced2, "vs. size:", A_reduced.shape[0])

# -------------------------------
# Step 6. Solve the Reduced System
# -------------------------------
# If the matrix is still singular, use the pseudo-inverse.
if rank_reduced2 < A_reduced.shape[0]:
    print("Matrix is singular; using pseudo-inverse for solution.")
    X_solution = np.linalg.pinv(A_reduced).dot(b_reduced)
else:
    X_solution = solve(A_reduced, b_reduced)

X_matrix = X_solution.reshape(N, J)
solution_df = pd.DataFrame(X_matrix, index=countries, columns=sectors)

print("Equilibrium Expenditures X^n_j:")
print(solution_df)
solution_df.to_csv(r'C:\Users\13424\PycharmProjects\ECON6903\equilibrium_expenditures.csv', index=True)
