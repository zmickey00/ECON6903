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
# Extract world value added (assumes header "VA_World")
world_va = va_world_df['VA_World'].iloc[0]
print("World Value Added =", world_va)

# Normalize deficits (divide by world value added)
deficits_df['Deficits'] = deficits_df['Deficits'] / world_va

# Build lists of countries and sectors.
countries = countries_df['Country'].tolist()
sectors = sorted(alpha_df['sector'].unique().tolist())

N = len(countries)  # Number of countries
J = len(sectors)  # Number of sectors
size = N * J  # Total number of unknowns


# Helper: map (country index, sector index) to vector index.
def idx(n, j):
    return n * J + j


# -------------------------------
# Step 2. Precompute Dictionaries for Fast Lookups
# -------------------------------
# alpha: key = (country, sector)
alpha_dict = {(row['country'], row['sector']): row['alpha']
              for _, row in alpha_df.iterrows()}

# deficits: key = country
deficit_dict = {row['country']: row['Deficits']
                for _, row in deficits_df.iterrows()}

# gamma_IO: key = (country, SectorOrigin, SectorDestination)
gamma_io_dict = {(row['country'], row['SectorOrigin'], row['SectorDestination']): row['gamma_IO']
                 for _, row in gamma_io_df.iterrows()}

# gamma_VA: key = (country, sector)
gamma_va_dict = {(row['country'], row['sector']): row['gamma_VA']
                 for _, row in gamma_va_df.iterrows()}

# one_plus_tau: key = (CountryOrigin, CountryDestination, sector)
one_plus_tau_dict = {(row['CountryOrigin'], row['CountryDestination'], row['sector']): row['one_plus_tau']
                     for _, row in one_plus_tau_df.iterrows()}

# For trade contributions, pi: key = (Origin, Destination, Sector)
pi_dict = {(row['CountryOrigin'], row['CountryDestination'], row['sector']): row['pi']
           for _, row in pi_df.iterrows()}
# For tariff revenue, we assume the same keys but using columns "CountryOrigin", etc.
pi_tariff_dict = {(row['CountryOrigin'], row['CountryDestination'], row['sector']): row['pi']
                  for _, row in pi_df.iterrows() if 'CountryOrigin' in row}

# -------------------------------
# Step 3. Construct the Linear System A x = b
# -------------------------------
A = np.zeros((size, size))
b = np.zeros(size)

# Loop over each country and sector combination.
for n, country in enumerate(countries):
    for j, sector_j in enumerate(sectors):
        eq_idx = idx(n, j)
        # Identity term for X^n_j
        A[eq_idx, eq_idx] = 1.0

        # --- (i) Trade contributions using gamma_IO
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
                    A[eq_idx, idx(i, k)] -= coeff

        # --- (ii) Preference and labor share contributions
        alpha_val = alpha_dict.get((country, sector_j), 0.0)
        # (a) Contribution from gamma_VA (labor share)
        for j_prime, sector_jprime in enumerate(sectors):
            gamma_va_val = gamma_va_dict.get((country, sector_jprime), 0.0)
            for i, country_i in enumerate(countries):
                key_tau = (country, country_i, sector_jprime)
                key_pi = (country, country_i, sector_jprime)
                if key_tau in one_plus_tau_dict and key_pi in pi_dict:
                    tau_val = one_plus_tau_dict[key_tau]
                    pi_val = pi_dict[key_pi]
                    coeff = alpha_val * gamma_va_val * (pi_val / tau_val)
                    A[eq_idx, idx(i, j_prime)] -= coeff

        # (b) Tariff revenue contribution (R_n)
        for j_prime, sector_jprime in enumerate(sectors):
            inner_sum = 0.0
            for i, country_i in enumerate(countries):
                key_tau = (country, country_i, sector_jprime)
                key_pi = (country, country_i, sector_jprime)
                if key_tau in one_plus_tau_dict and key_pi in pi_tariff_dict:
                    base_tau = one_plus_tau_dict[key_tau]
                    # Tariff rate τ = one_plus_tau - 1.
                    tau_val = base_tau - 1.0
                    pi_val = pi_tariff_dict[key_pi]
                    inner_sum += (tau_val * pi_val) / base_tau
            coeff = alpha_val * inner_sum
            # This revenue contribution only affects the domestic sector (n, j_prime)
            A[eq_idx, idx(n, j_prime)] -= coeff

        # --- (iii) Set right-hand side from deficits.
        deficit_val = deficit_dict.get(country, 0.0)
        b[eq_idx] = alpha_val * deficit_val

# -------------------------------
# Step 4. Impose Normalization Condition
# -------------------------------
# Replace the last equation with: ∑ₙ (w_n L_n) = 1, approximated by summing gamma_VA.
norm_row = size - 1
A[norm_row, :] = 0  # Clear last row
for n, country in enumerate(countries):
    for j, sector in enumerate(sectors):
        gamma_va_val = gamma_va_dict.get((country, sector), 0.0)
        A[norm_row, idx(n, j)] = gamma_va_val
b[norm_row] = 1.0

# -------------------------------
# Step 5. Solve the System and Output the Result
# -------------------------------
# X_solution = solve(A, b)
# X_matrix = X_solution.reshape(N, J)
# solution_df = pd.DataFrame(X_matrix, index=countries, columns=sectors)
# rank = np.linalg.matrix_rank(A)
# print("Matrix rank:", rank, " vs. size:", size)

# print("Equilibrium Expenditures X^n_j:")
# print(solution_df)




# Step 5: Drop one redundant equation
drop_row = size - 1  # Drop the last equation (market clearing)
A_reduced = np.delete(A, drop_row, axis=0)
b_reduced = np.delete(b, drop_row, axis=0)
# Solve the reduced system
X_solution = solve(A_reduced, b_reduced)
X_matrix = X_solution.reshape(N, J)
solution_df = pd.DataFrame(X_matrix, index=countries, columns=sectors)
print("Matrix rank:", np.linalg.matrix_rank(A_reduced), "vs. size:", A_reduced.shape[0])
print("Equilibrium Expenditures X^n_j:")
print(solution_df)
