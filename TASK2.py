import pandas as pd
import numpy as np
from numpy.linalg import inv

# ===============================
# Step 0. Import Data Files and Clean Data
# ===============================

alpha = pd.read_csv("alpha.csv")  # Expected columns: 'country', 'sector', 'alpha'
countries_df = pd.read_csv("CountryNames.csv")  # Expected columns: 'Country'
deficits = pd.read_csv("Deficits.csv")  # Expected columns: 'country', 'Deficits'
gamma_io = pd.read_csv("gamma_IO.csv")  # Expected columns: 'country', 'SectorOrigin', 'SectorDestination', 'gamma_IO'
gamma_va = pd.read_csv("gamma_VA.csv")  # Expected columns: 'country', 'sector', 'gamma_VA'
one_plus_tau = pd.read_csv(
    "one_plus_tau.csv")  # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'one_plus_tau'
pi_df = pd.read_csv("pi.csv")  # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'pi'
va_world = pd.read_csv("VA_World.csv")

# Clean column names and remove extra spaces
for df in [alpha, countries_df, deficits, gamma_io, gamma_va, one_plus_tau, pi_df]:
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

# Normalize deficits using world value added
world_va = va_world['VA_World'].iloc[0]  # Assuming a single value is provided
deficits['Deficits'] = deficits['Deficits'] / world_va

# Dimensions
J = 40  # number of sectors
N = 31  # number of countries

# ===============================
# Create Lookup Dictionaries
# ===============================
# For alpha, key: (sector, country)
alpha.columns = alpha.columns.str.strip()  # Remove extra spaces
alpha_dict = {(row['sector'], row['country']): row['alpha'] for _, row in alpha.iterrows()}

# For one_plus_tau, key: (sector, destination, origin)
one_plus_tau.columns = one_plus_tau.columns.str.strip()  # Remove extra spaces
one_plus_tau_dict = {(row['sector'], row['CountryDestination'], row['CountryOrigin']): row['one_plus_tau']
                     for _, row in one_plus_tau.iterrows()}

# For pi, key: (sector, destination, origin)
pi_df.columns = pi_df.columns.str.strip()  # Remove extra space
pi_dict = {(row['sector'], row['CountryDestination'], row['CountryOrigin']): row['pi']
           for _, row in pi_df.iterrows()}

# For gamma_io, key: (SectorOrigin, SectorDestination, country)
gamma_io.columns = gamma_io.columns.str.strip()  # Remove extra space
gamma_io_dict = {(row['SectorOrigin'], row['SectorDestination'], row['country']): row['gamma_IO']
                 for _, row in gamma_io.iterrows()}

# For gamma_va, key: (sector, country)
gamma_va.columns = gamma_va.columns.str.strip()  # Remove extra space
gamma_va_dict = {(row['sector'], row['country']): row['gamma_VA']
                 for _, row in gamma_va.iterrows()}

# For deficits, key: country
deficits.columns = deficits.columns.str.strip()  # Remove extra space
deficits_dict = {row['country']: row['Deficits'] for _, row in deficits.iterrows()}


# ===============================
# Define Lookup Functions Using the Dictionaries
# ===============================
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


# ===============================
# Other Helper Functions (for the market clearing system)
# ===============================
def sum_last(j, n, k):
    total = 0
    # Loop over origin country index (1-based)
    for index in range(1, N + 1):
        total += (read_alpha(j, n) *
                  (one_plus_tao(k, n, index) - 1) *
                  read_pi(k, n, index)) / one_plus_tao(k, n, index)
    return total


def big_i(a, b):
    return 1 if a == b else 0


def item_3(n, i, j, k):
    return big_i(n, i) * sum_last(j, n, k)


def item_2(n, i, j, k):
    return (read_pi(k, i, n) / one_plus_tao(k, i, n)) * (read_gama_1(j, k, n) + read_alpha(j, n) * read_gama_2(k, n))


def item_1(n, i, j, k):
    return big_i((j - 1) * N + n, (k - 1) * N + i)


def m_entry(n, i, j, k):
    return item_1(n, i, j, k) - item_2(n, i, j, k) - item_3(n, i, j, k)


# ===============================
# Task 1: Solve for the Initial Equilibrium Expenditures
# ===============================
def solve_initial_equilibrium_fixed_point():
    """
    Solve for initial equilibrium expenditures X_n^j by writing the system in long form and solving Ax = b.
    """
    # Build matrix M (dimensions: J*N+1 rows, J*N columns)
    M = np.zeros((J * N + 1, J * N))
    for j in range(J):  # 0-based loop for sectors; sector = j+1
        for n in range(N):  # 0-based loop for country; country = n+1
            for k in range(J):  # 0-based loop for sectors; sector = k+1
                for i in range(N):  # 0-based loop for country; country = i+1
                    m_row = j * N + n
                    m_col = k * N + i
                    M[m_row, m_col] = m_entry(n + 1, i + 1, j + 1, k + 1)

    # Build vector B (dimensions: J*N+1 x 1)
    B = np.zeros((J * N + 1, 1))
    for j in range(J):
        for n in range(N):
            b_row = j * N + n
            B[b_row, 0] = read_alpha(j + 1, n + 1) * read_d(n + 1)

    # Replace the last row of M with the normalization row:
    # For each column m corresponding to (k,i), set:
    # M[J*N, m] = sum_{n=1}^{N} [ read_gama_2(k, n)* read_pi(k, i, n)/one_plus_tao(k, i, n) ]
    for k in range(J):
        for i in range(N):
            m_col = k * N + i
            M[J * N, m_col] = sum(
                read_gama_2(k + 1, n + 1) * read_pi(k + 1, i + 1, n + 1) / one_plus_tao(k + 1, i + 1, n + 1)
                for n in range(N))
    # Set the corresponding entry in B for the normalization row to 1.
    B[J * N, 0] = 1

    # Remove one redundant equation (remove the last row) to create a square system.
    M_reduced = np.delete(M, J * N-1, axis=0)
    B_reduced = np.delete(B, J * N-1, axis=0)

    # Solve for X (in long form).
    X_vector = np.dot(inv(M_reduced), B_reduced)


    # Convert the long vector X into a nested dictionary X[j][n]
    X = {j: {} for j in range(1, J + 1)}
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            index = (j - 1) * N + (n - 1)
            X[j][n] = X_vector[index, 0]

    # Save results into a DataFrame and export to CSV.
    X_matrix = np.zeros((J, N))
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            X_matrix[j - 1, n - 1] = X[j][n]
    country_names_list = countries_df['Country'].tolist()
    sector_names = [f"Sector_{j}" for j in range(1, J + 1)]
    result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_names_list)
    result_df.to_csv('initial_equilibrium_expenditures.csv', index=True)
    print("Task 1: Initial equilibrium expenditures saved to 'initial_equilibrium_expenditures.csv'")

    return X, {'alpha': alpha, 'CountryNames': countries_df, 'deficits': deficits,
               'gamma_IO': gamma_io, 'gamma_VA': gamma_va, 'one_plus_tau': one_plus_tau, 'pi': pi_df}, result_df


# ===============================
# Task 2: Counterfactual Experiments
# (Following Caliendo and Parro (2015))
# ===============================
THETA = 4


def counterfactual_experiment(case=1, epsilon=0.1, max_iter_outer=200, tol_outer=1e-8):
    """
    Solve for the counterfactual equilibrium.

    Parameters:
      case: (int)
         1: Baseline counterfactual (𝜆̂ = κ̂ = 1, deficits unchanged)
         2: Eliminate trade imbalances (set D'_n = 0)
         3: Autarky approximation (for n ≠ i, set κ̂_{ni}^j = 30 and D'_n = 0)
         4: Productivity shock in China (set 𝜆̂_{China}^j = 5 for all sectors; assume China is index 7)
      epsilon: damping factor for outer loop
      max_iter_outer: maximum iterations for wage updating
      tol_outer: convergence tolerance for wages

    Returns:
      w_hat: final counterfactual wage changes (dictionary)
      X_new: counterfactual expenditure distribution (nested dictionary)
      pi_new: updated trade shares (nested dictionary)
    """
    print(f"\nExecuting counterfactual experiment Case {case}...")
    theta = THETA

    # Load initial equilibrium and data.
    X_initial, data, _ = solve_initial_equilibrium_fixed_point()
    countries_df = data['CountryNames']
    N = len(countries_df)
    J = len(data['alpha']['sector'].unique())

    # Create lookup dictionaries.
    alpha_dict = {(row['sector'], row['country']): row['alpha'] for _, row in data['alpha'].iterrows()}
    gamma_io_dict = {(row['SectorOrigin'], row['SectorDestination'], row['country']): row['gamma_IO'] for _, row in
                     data['gamma_IO'].iterrows()}
    gamma_va_dict = {(row['sector'], row['country']): row['gamma_VA'] for _, row in data['gamma_VA'].iterrows()}
    pi_dict = {(row['sector'], row['CountryDestination'], row['CountryOrigin']): row['pi'] for _, row in
               data['pi'].iterrows()}
    one_plus_tau_dict = {(row['sector'], row['CountryDestination'], row['CountryOrigin']): row['one_plus_tau'] for
                         _, row in data['one_plus_tau'].iterrows()}
    deficits_dict = {row['country']: row['Deficits'] for _, row in data['deficits'].iterrows()}

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

    # Set baseline counterfactual parameters.
    lambda_hat = {j: {i: 1.0 for i in range(1, N + 1)} for j in range(1, J + 1)}
    kappa_hat = {j: {n: {i: 1.0 for i in range(1, N + 1)} for n in range(1, N + 1)} for j in range(1, J + 1)}
    deficits_new = {n: deficits_dict.get(n, 0.0) for n in range(1, N + 1)}

    if case == 2:
        deficits_new = {n: 0.0 for n in range(1, N + 1)}
    elif case == 3:
        deficits_new = {n: 0.0 for n in range(1, N + 1)}
        for j in range(1, J + 1):
            for n in range(1, N + 1):
                for i in range(1, N + 1):
                    kappa_hat[j][n][i] = 30.0 if n != i else 1.0
    elif case == 4:
        for j in range(1, J + 1):
            lambda_hat[j][7] = 5.0

    # Initialize wage changes: w_hat
    w_hat = {n: 1.0 for n in range(1, N + 1)}

    # Outer loop: update wages.
    for iter_outer in range(max_iter_outer):
        # Inner loop: update price indices P_hat.
        P_hat = {j: {n: 1.0 for n in range(1, N + 1)} for j in range(1, J + 1)}
        max_iter_inner = 100
        tol_inner = 1e-6
        for iter_inner in range(max_iter_inner):
            c_hat = {}
            for j in range(1, J + 1):
                c_hat[j] = {}
                for n in range(1, N + 1):
                    gamma_final = read_gama_2(j, n)
                    cost = w_hat[n] ** gamma_final
                    for k in range(1, J + 1):
                        gamma_inter = read_gama_1(j, k, n)
                        cost *= P_hat[k][n] ** gamma_inter
                    c_hat[j][n] = cost
            P_hat_new = {}
            for j in range(1, J + 1):
                P_hat_new[j] = {}
                for n in range(1, N + 1):
                    sum_term = 0.0
                    for i in range(1, N + 1):
                        pi_val = read_pi(j, n, i)
                        lambda_val = lambda_hat[j][i]
                        barrier = kappa_hat[j][n][i]
                        sum_term += pi_val * lambda_val * (c_hat[j][i] * barrier) ** (-theta)
                    P_hat_new[j][n] = sum_term ** (-1 / theta) if sum_term > 0 else P_hat[j][n]
            diff_inner = max(abs(P_hat_new[j][n] - P_hat[j][n])
                             for j in range(1, J + 1) for n in range(1, N + 1))
            P_hat = P_hat_new
            if diff_inner < tol_inner:
                break

        # Step 4: Update trade shares.
        pi_hat = {}
        for j in range(1, J + 1):
            pi_hat[j] = {}
            for n in range(1, N + 1):
                pi_hat[j][n] = {}
                for i in range(1, N + 1):
                    pi_hat[j][n][i] = lambda_hat[j][i] * ((c_hat[j][i] * kappa_hat[j][n][i]) / P_hat[j][n]) ** (-theta)
        pi_new = {}
        for j in range(1, J + 1):
            pi_new[j] = {}
            for n in range(1, N + 1):
                pi_new[j][n] = {}
                for i in range(1, N + 1):
                    pi_new[j][n][i] = read_pi(j, n, i) * pi_hat[j][n][i]

        # Step 5: Solve for new expenditures X_new using market clearing.
        X_new = {j: {n: X_initial[j][n] for n in range(1, N + 1)} for j in range(1, J + 1)}
        max_iter_income = 100
        tol_income = 1e-8
        for iter_income in range(max_iter_income):
            wage_income_new = {n: 0.0 for n in range(1, N + 1)}
            tariff_revenue_new = {n: 0.0 for n in range(1, N + 1)}
            for n in range(1, N + 1):
                for j in range(1, J + 1):
                    for i in range(1, N + 1):
                        gamma_final = read_gama_2(j, n)
                        pi_val_new = pi_new.get(j, {}).get(i, {}).get(n, 0.0)
                        tau_val = one_plus_tao(j, i, n)
                        wage_income_new[n] += gamma_final * X_new[j][i] * pi_val_new / tau_val
                for j in range(1, J + 1):
                    for i in range(1, N + 1):
                        tau_val = one_plus_tao(j, n, i)
                        pi_val_new = pi_new.get(j, {}).get(n, {}).get(i, 0.0)
                        if tau_val > 1.0:
                            tariff_revenue_new[n] += (tau_val - 1.0) / tau_val * X_new[j][n] * pi_val_new
            income_new = {n: wage_income_new[n] + tariff_revenue_new[n] + deficits_new.get(n, 0.0)
                          for n in range(1, N + 1)}
            X_new_upd = {j: {n: 0.0 for n in range(1, N + 1)} for j in range(1, J + 1)}
            for j in range(1, J + 1):
                for n in range(1, N + 1):
                    intermediate_demand = 0.0
                    for k in range(1, J + 1):
                        for i in range(1, N + 1):
                            gamma_inter = read_gama_1(j, k, n)
                            pi_val_new = pi_new.get(k, {}).get(i, {}).get(n, 0.0)
                            tau_val = one_plus_tao(k, i, n)
                            intermediate_demand += gamma_inter * X_new[k][i] * pi_val_new / tau_val
                    final_demand = read_alpha(j, n) * income_new[n]
                    X_new_upd[j][n] = intermediate_demand + final_demand
            total_va = sum(wage_income_new.values())
            scaling_factor = 1.0 / total_va if total_va > 0 else 1.0
            for j in range(1, J + 1):
                for n in range(1, N + 1):
                    X_new_upd[j][n] *= scaling_factor
            diff_income = max(abs(X_new_upd[j][n] - X_new[j][n])
                              for j in range(1, J + 1) for n in range(1, N + 1))
            X_new = X_new_upd
            if diff_income < tol_income:
                break

        # Step 6: Update wages.
        w_hat_new = {}
        for n in range(1, N + 1):
            numerator = 0.0
            for j in range(1, J + 1):
                for i in range(1, N + 1):
                    gamma_final = read_gama_2(j, n)
                    pi_val_new = pi_new.get(j, {}).get(i, {}).get(n, 0.0)
                    tau_val = one_plus_tao(j, i, n)
                    numerator += gamma_final * X_new[j][i] * pi_val_new / tau_val
            old_income = 0.0
            for j in range(1, J + 1):
                for i in range(1, N + 1):
                    gamma_final = read_gama_2(j, n)
                    pi_val_old = read_pi(j, i, n)
                    tau_val = one_plus_tao(j, i, n)
                    old_income += gamma_final * X_initial[j][i] * pi_val_old / tau_val
            w_hat_new[n] = numerator / old_income if old_income > 0 else w_hat[n]
        diff_outer = max(abs(w_hat_new[n] - w_hat[n]) for n in range(1, N + 1))
        for n in range(1, N + 1):
            w_hat[n] = (1 - epsilon) * w_hat[n] + epsilon * w_hat_new[n]
        print(f"Outer iteration {iter_outer}: max wage diff = {diff_outer:.8f}")
        if diff_outer < tol_outer:
            print(f"Convergence achieved after {iter_outer + 1} outer iterations.")
            break

    return w_hat, X_new, pi_new


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    # Task 1: Solve for initial equilibrium expenditures.
    X_initial, data, result_df = solve_initial_equilibrium_fixed_point()
    print("\nTask 1 completed. (First 5 sectors x 5 countries):")
    print(result_df.iloc[:5, :5])

    # Task 2: Run a counterfactual experiment.
    # Change the argument case=... to try different counterfactual scenarios.
    w_hat, X_new, pi_new = counterfactual_experiment(case=2)
    print("\nTask 2 completed. Counterfactual wage changes w_hat:")
    print(w_hat)
