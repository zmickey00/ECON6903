import pandas as pd
import numpy as np
from numpy.linalg import solve

# -------------------------------
# Step 0. Import Data Files
# -------------------------------
# Adjust file paths if your files are in a subfolder (e.g., 'data/')
alpha         = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\alpha.csv')         # Expected columns: 'Country', 'Sector', 'alpha'
countries_df  = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\CountryNames.csv')    # Expected columns: 'Country' (or similar)
deficits      = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\Deficits.csv')        # Expected columns: 'Country', 'Deficit'
gamma_io      = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\gamma_IO.csv')        # Expected columns: 'Country', 'FromSector', 'ToSector', 'gamma'
gamma_va      = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\gamma_VA.csv')        # Expected columns: 'Country', 'Sector', 'gamma_VA'
one_plus_tau  = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\one_plus_tau.csv')    # Expected columns: 'Origin', 'Destination', 'Sector', 'one_plus_tau'
pi_df         = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\pi.csv')              # Expected columns: 'Origin', 'Destination', 'Sector', 'pi'
va_world      = pd.read_csv(r'C:\Users\13424\PycharmProjects\ECON6903\data\VA_World.csv')        # Expected column: 'VA_World'

# -------------------------------
# Step 1. Data Preprocessing
# -------------------------------
# Print columns to verify (uncomment if needed)
#print("alpha:", alpha.columns)
#print("CountryNames:", countries_df.columns)
#print("Deficits:", deficits.columns)
#print("gamma_IO:", gamma_io.columns)
#print("gamma_VA:", gamma_va.columns)
#print("one_plus_tau:", one_plus_tau.columns)
#print("pi:", pi_df.columns)
#print("VA_World:", va_world.columns)

# For VA_World, the header is "VA_World"
world_va = va_world['VA_World'].iloc[0]
print("World Value Added =", world_va)

# Normalize deficits (divide by world value added)
deficits['Deficits'] = deficits['Deficits'] / world_va

# Create a list of countries using the country names CSV.
countries = countries_df['Country'].tolist()

# Assume that the sectors are uniquely listed in alpha.csv.
sectors = sorted(alpha['sector'].unique().tolist())

# Determine dimensions: N = number of countries, J = number of sectors.
N = len(countries)
J = len(sectors)

# Define an index helper to map (country, sector) pairs to vector indices.
def idx(n, j):
    return n * J + j

size = N * J

# -------------------------------
# Step 2. Construct the Linear System
# -------------------------------
# We will set up a system A x = b where x is the vectorized form of X_j^n.
# The following loop is an illustrative example. You must verify that the algebra matches equations (4)-(7)
# with the normalization (8) replacing one redundant equation.

A = np.zeros((size, size))
b = np.zeros(size)

# Loop over each country (n) and sector (j)
for n in range(N):
    for j in range(J):
        eq_idx = idx(n, j)
        # Start with an identity coefficient for X_j^n
        A[eq_idx, eq_idx] = 1.0

        # --- (i) Subtract trade contributions:
        # Loop over sectors k (destination sector in the trade network) and countries i (origin)
        for k in range(J):
            # Fetch the gamma_IO parameter for trade from sector j to sector k in country n.
            # (Assumes gamma_IO file has column 'gamma')
            gamma_io_rows = gamma_io[
                (gamma_io['country'] == countries[n]) &
                (gamma_io['SectorOrigin'] == sectors[j]) &
                (gamma_io['SectorDestination'] == sectors[k])
            ]
            if gamma_io_rows.empty:
                continue
            gamma_val = gamma_io_rows['gamma'].iloc[0]
            for i in range(N):
                # Get one_plus_tau and pi for trade from country i to country n in sector k.
                tau_rows = one_plus_tau[
                    (one_plus_tau['CountryOrigin'] == countries[i]) &
                    (one_plus_tau['CountryDestination'] == countries[n]) &
                    (one_plus_tau['sector'] == sectors[k])
                ]
                pi_rows = pi_df[
                    (pi_df['Origin'] == countries[i]) &
                    (pi_df['Destination'] == countries[n]) &
                    (pi_df['Sector'] == sectors[k])
                ]
                if tau_rows.empty or pi_rows.empty:
                    continue
                tau_val = tau_rows['one_plus_tau'].iloc[0]
                pi_val = pi_rows['pi'].iloc[0]
                coeff = gamma_val * (pi_val / tau_val)
                A[eq_idx, idx(i, k)] -= coeff

        # --- (ii) Subtract preference terms:
        # Get the alpha parameter from alpha.csv for country n and sector j.
        alpha_row = alpha[
            (alpha['country'] == countries[n]) &
            (alpha['sector'] == sectors[j])
        ]
        if not alpha_row.empty:
            alpha_val = alpha_row['alpha'].iloc[0]
        else:
            alpha_val = 0.0

        # (a) Contribution from the labor share (gamma_VA) term.
        for j_prime in range(J):
            gamma_va_rows = gamma_va[
                (gamma_va['country'] == countries[n]) &
                (gamma_va['sector'] == sectors[j_prime])
            ]
            if gamma_va_rows.empty:
                continue
            gamma_va_val = gamma_va_rows['gamma_VA'].iloc[0]
            for i in range(N):
                tau_rows = one_plus_tau[
                    (one_plus_tau['Origin'] == countries[n]) &
                    (one_plus_tau['Destination'] == countries[i]) &
                    (one_plus_tau['Sector'] == sectors[j_prime])
                ]
                pi_rows = pi_df[
                    (pi_df['Origin'] == countries[n]) &
                    (pi_df['Destination'] == countries[i]) &
                    (pi_df['Sector'] == sectors[j_prime])
                ]
                if tau_rows.empty or pi_rows.empty:
                    continue
                tau_val = tau_rows['one_plus_tau'].iloc[0]
                pi_val = pi_rows['pi'].iloc[0]
                coeff = alpha_val * gamma_va_val * (pi_val / tau_val)
                A[eq_idx, idx(i, j_prime)] -= coeff

        # (b) Contribution from R_n (tariff revenue term):
        for j_prime in range(J):
            inner_sum = 0.0
            for i in range(N):
                tau_rows = one_plus_tau[
                    (one_plus_tau['CountryOrigin'] == countries[n]) &
                    (one_plus_tau['CountryDestination'] == countries[i]) &
                    (one_plus_tau['sector'] == sectors[j_prime])
                ]
                pi_rows = pi_df[
                    (pi_df['CountryOrigin'] == countries[n]) &
                    (pi_df['CountryDestination'] == countries[i]) &
                    (pi_df['sector'] == sectors[j_prime])
                ]
                if tau_rows.empty or pi_rows.empty:
                    continue
                # Compute tariff rate: since one_plus_tau = 1 + τ.
                tau_val = tau_rows['one_plus_tau'].iloc[0] - 1.0
                pi_val = pi_rows['pi'].iloc[0]
                inner_sum += (tau_val * pi_val) / tau_rows['one_plus_tau'].iloc[0]
            coeff = alpha_val * inner_sum
            A[eq_idx, idx(n, j_prime)] -= coeff

        # --- (iii) Set the right-hand side from the deficit contribution.
        deficit_row = deficits[deficits['country'] == countries[n]]
        if not deficit_row.empty:
            deficit_val = deficit_row['Deficits'].iloc[0]
        else:
            deficit_val = 0.0
        b[eq_idx] = alpha_val * deficit_val

# -------------------------------
# Step 3. Impose Normalization Condition
# -------------------------------
# Replace one of the equations with the normalization condition:
# ∑ₙ (w_n L_n) = 1. We approximate w_nL_n using the gamma_VA values.
norm_row = size - 1
A[norm_row, :] = 0  # Clear the normalization row
for n in range(N):
    for j in range(J):
        gamma_va_rows = gamma_va[
            (gamma_va['country'] == countries[n]) &
            (gamma_va['sector'] == sectors[j])
        ]
        if gamma_va_rows.empty:
            coeff = 0.0
        else:
            coeff = gamma_va_rows['gamma_VA'].iloc[0]
        A[norm_row, idx(n, j)] = coeff
b[norm_row] = 1.0

# -------------------------------
# Step 4. Solve the Linear System
# -------------------------------
X_solution = solve(A, b)

# Reshape the solution vector into a (countries x sectors) DataFrame for clarity.
X_matrix = X_solution.reshape(N, J)
solution_df = pd.DataFrame(X_matrix, index=countries, columns=sectors)

print("Equilibrium Expenditures X^n_j:")
print(solution_df)
