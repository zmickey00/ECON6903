import pandas as pd
import numpy as np
from numpy.linalg import inv


# ===============================
# Helper Function: Create Map from DataFrame
# ===============================
def create_map_from_df(df, *keys, value_col):
    """
    Creates a nested dictionary (map) from a DataFrame.
    For each key column, casts the value to int.
    """
    result = {}
    for _, row in df.iterrows():
        curr = result
        for key in keys[:-1]:
            key_val = int(row[key])
            if key_val not in curr:
                curr[key_val] = {}
            curr = curr[key_val]
        last_key = int(row[keys[-1]])
        curr[last_key] = float(row[value_col])
    return result


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
world_va = va_world['VA_World'].iloc[0]
deficits['Deficits'] = deficits['Deficits'] / world_va

# Dimensions
J = 40  # number of sectors
N = 31  # number of countries

# ===============================
# Create Lookup Dictionaries (with numeric keys)
# ===============================
alpha_map = create_map_from_df(alpha, 'sector', 'country', value_col='alpha')
one_plus_tau_map = create_map_from_df(one_plus_tau, 'sector', 'CountryDestination', 'CountryOrigin',
                                      value_col='one_plus_tau')
pi_map = create_map_from_df(pi_df, 'sector', 'CountryDestination', 'CountryOrigin', value_col='pi')
gamma_io_map = create_map_from_df(gamma_io, 'SectorOrigin', 'SectorDestination', 'country', value_col='gamma_IO')
gamma_va_map = create_map_from_df(gamma_va, 'sector', 'country', value_col='gamma_VA')
# For deficits, we assume the CSV uses numeric codes for country.
deficits_dict = {int(row['country']): float(row['Deficits']) for _, row in deficits.iterrows()}


# ===============================
# Define Lookup Functions Using the Dictionaries
# ===============================
def read_alpha(sector, country):
    return alpha_map[int(sector)][int(country)]


def one_plus_tao(sector, destination, origin):
    return one_plus_tau_map[int(sector)][int(destination)][int(origin)]


def read_pi(sector, destination, origin):
    return pi_map[int(sector)][int(destination)][int(origin)]


def read_gama_1(sorigin, sdestination, country):
    return gamma_io_map[int(sorigin)][int(sdestination)][int(country)]


def read_gama_2(sector, country):
    return gamma_va_map[int(sector)][int(country)]


def read_d(country):
    # Here country is expected to be an integer (or convertible to int).
    return deficits_dict[int(country)]


# ===============================
# Other Helper Functions (for the market clearing system)
# ===============================
def sum_last(j, n, k):
    total = 0
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
    Solve for initial equilibrium expenditures Xₙʲ by writing the system in long form and solving Ax = b.
    """
    M = np.zeros((J * N + 1, J * N))
    for j in range(J):  # sectors: 0,...,J-1 (sector = j+1)
        for n in range(N):  # countries: 0,...,N-1 (country = n+1)
            for k in range(J):
                for i in range(N):
                    m_row = j * N + n
                    m_col = k * N + i
                    M[m_row, m_col] = m_entry(n + 1, i + 1, j + 1, k + 1)

    B = np.zeros((J * N + 1, 1))
    # Use numeric country codes (n+1) when calling read_d.
    for j in range(J):
        for n in range(N):
            b_row = j * N + n
            B[b_row, 0] = read_alpha(j + 1, n + 1) * read_d(n + 1)
    for k in range(J):
        for i in range(N):
            m_col = k * N + i
            M[J * N, m_col] = sum(
                read_gama_2(k + 1, n + 1) * read_pi(k + 1, i + 1, n + 1) / one_plus_tao(k + 1, i + 1, n + 1)
                for n in range(N))
    B[J * N, 0] = 1

    M_reduced = np.delete(M, J * N - 1, axis=0)
    B_reduced = np.delete(B, J * N - 1, axis=0)

    X_vector = np.dot(inv(M_reduced), B_reduced)

    X = {j: {} for j in range(1, J + 1)}
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            index = (j - 1) * N + (n - 1)
            X[j][n] = X_vector[index, 0]

    X_matrix = np.zeros((J, N))
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            X_matrix[j - 1, n - 1] = X[j][n]
    country_list = [str(i) for i in range(1, N + 1)]  # Using numeric country codes as strings.
    sector_names = [f"Sector_{j}" for j in range(1, J + 1)]
    result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_list)
    result_df.to_csv('initial_equilibrium_expenditures.csv', index=True)
    print("Task 1: Initial equilibrium expenditures saved to 'initial_equilibrium_expenditures.csv'")

    return X, {'alpha': alpha, 'CountryNames': countries_df, 'deficits': deficits,
               'gamma_IO': gamma_io, 'gamma_VA': gamma_va, 'one_plus_tau': one_plus_tau, 'pi': pi_df}, result_df


# ===============================
# Task 2: Counterfactual Experiments (Following Caliendo and Parro (2015))
# ===============================
THETA = 4


def counterfactual_experiment(case=1, epsilon=0.1, max_iter_outer=100, tol_outer=1e-6):
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

    X_initial, data, _ = solve_initial_equilibrium_fixed_point()
    countries_df = data['CountryNames']
    # For Task 2, we assume country codes are numeric, so create a list of codes 1,2,...,N.
    N = 31
    country_list = list(range(1, N + 1))
    N = len(country_list)
    J = len(data['alpha']['sector'].unique())


    alpha_map = create_map_from_df(data['alpha'], 'sector', 'country', value_col='alpha')
    gamma_io_map = create_map_from_df(data['gamma_IO'], 'SectorOrigin', 'SectorDestination', 'country',
                                      value_col='gamma_IO')
    gamma_va_map = create_map_from_df(data['gamma_VA'], 'sector', 'country', value_col='gamma_VA')
    pi_map = create_map_from_df(data['pi'], 'sector', 'CountryDestination', 'CountryOrigin', value_col='pi')
    one_plus_tau_map = create_map_from_df(data['one_plus_tau'], 'sector', 'CountryDestination', 'CountryOrigin',
                                          value_col='one_plus_tau')
    deficits_map = {int(row['country']): float(row['Deficits']) for _, row in data['deficits'].iterrows()}

    def read_alpha(sector, country):
        return alpha_map[int(sector)][int(country)]

    def one_plus_tao(sector, destination, origin):
        return one_plus_tau_map[int(sector)][int(destination)][int(origin)]

    def read_pi(sector, destination, origin):
        return pi_map[int(sector)][int(destination)][int(origin)]

    def read_gama_1(sorigin, sdestination, country):
        return gamma_io_map[int(sorigin)][int(sdestination)][int(country)]

    def read_gama_2(sector, country):
        return gamma_va_map[int(sector)][int(country)]

    def read_d(country):
        return deficits_map[int(country)]

    lambda_hat = {j: {i: 1.0 for i in range(1, N + 1)} for j in range(1, J + 1)}
    kappa_hat = {j: {n: {i: 1.0 for i in range(1, N + 1)} for n in range(1, N + 1)} for j in range(1, J + 1)}
    deficits_new = {country_list[i - 1]: deficits_map.get(i, 0.0) for i in range(1, N + 1)}
    if case == 2:
        deficits_new = {country_list[i - 1]: 0.0 for i in range(1, N + 1)}
    elif case == 3:
        deficits_new = {country_list[i - 1]: 0.0 for i in range(1, N + 1)}
        for j in range(1, J + 1):
            for n in range(1, N + 1):
                for i in range(1, N + 1):
                    kappa_hat[j][n][i] = 30.0 if n != i else 1.0
    elif case == 4:
        for j in range(1, J + 1):
            lambda_hat[j][7] = 5.0

    w_hat = {n: 1.0 for n in range(1, N + 1)}
    w_hat_arr = np.ones(N)

    # Convert lambda_hat and kappa_hat to arrays.
    lambda_arr = np.empty((J, N))
    for j in range(1, J + 1):
        for i in range(1, N + 1):
            lambda_arr[j - 1, i - 1] = lambda_hat[j][i]
    kappa_arr = np.empty((J, N, N))
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            for i in range(1, N + 1):
                kappa_arr[j - 1, n - 1, i - 1] = kappa_hat[j][n][i]

    # Convert pi and one_plus_tau to arrays.
    pi_arr = np.empty((J, N, N))
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            for i in range(1, N + 1):
                pi_arr[j - 1, n - 1, i - 1] = read_pi(j, n, i)
    tau_arr = np.empty((J, N, N))
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            for i in range(1, N + 1):
                tau_arr[j - 1, n - 1, i - 1] = one_plus_tao(j, n, i)

    # Convert gamma_va and gamma_io to arrays.
    gamma_va_arr = np.empty((J, N))
    for j in range(1, J + 1):
        for n in range(1, N + 1):
            gamma_va_arr[j - 1, n - 1] = read_gama_2(j, n)
    gamma_io_arr = np.empty((J, J, N))
    for j in range(1, J + 1):
        for k in range(1, J + 1):
            for n in range(1, N + 1):
                gamma_io_arr[j - 1, k - 1, n - 1] = read_gama_1(j, k, n)

    from numba import njit, prange

    @njit(parallel=True)
    def update_c_hat(w_hat_arr, P_hat, gamma_va_arr, gamma_io_arr, J, N):
        c_hat = np.empty((J, N))
        for j in prange(J):
            for n in range(N):
                cost = w_hat_arr[n] ** gamma_va_arr[j, n]
                for k in range(J):
                    cost *= P_hat[k, n] ** gamma_io_arr[j, k, n]
                c_hat[j, n] = cost
        return c_hat

    @njit(parallel=True)
    def update_P_hat(c_hat, pi_arr, lambda_arr, kappa_arr, theta, J, N, P_hat_old):
        P_hat_new = np.empty((J, N))
        for j in prange(J):
            for n in range(N):
                sum_term = 0.0
                for i in range(N):
                    sum_term += pi_arr[j, n, i] * lambda_arr[j, i] * (c_hat[j, i] * kappa_arr[j, n, i]) ** (-theta)
                if sum_term > 0:
                    P_hat_new[j, n] = sum_term ** (-1.0 / theta)
                else:
                    P_hat_new[j, n] = P_hat_old[j, n]
        return P_hat_new

    P_hat = np.ones((J, N))
    for iter_outer in range(max_iter_outer):
        for iter_inner in range(100):
            c_hat = update_c_hat(np.array(list(w_hat.values())), P_hat, gamma_va_arr, gamma_io_arr, J, N)
            P_hat_new = update_P_hat(c_hat, pi_arr, lambda_arr, kappa_arr, theta, J, N, P_hat)
            diff_inner = np.max(np.abs(P_hat_new - P_hat))
            P_hat = P_hat_new
            if diff_inner < 1e-6:
                break

        wage_income_new = {n: 0.0 for n in range(1, N + 1)}
        tariff_revenue_new = {n: 0.0 for n in range(1, N + 1)}
        X_new = {j: {n: X_initial[j][n] for n in range(1, N + 1)} for j in range(1, J + 1)}
        max_iter_income = 100
        tol_income = 1e-8
        for iter_income in range(max_iter_income):
            for n in range(1, N + 1):
                for j in range(1, J + 1):
                    for i in range(1, N + 1):
                        gamma_final = read_gama_2(j, n)
                        pi_val_new = read_pi(j, n, i)
                        tau_val = one_plus_tao(j, i, n)
                        wage_income_new[n] += gamma_final * X_new[j][i] * pi_val_new / tau_val
                for j in range(1, J + 1):
                    for i in range(1, N + 1):
                        tau_val = one_plus_tao(j, n, i)
                        pi_val_new = read_pi(j, n, i)
                        if tau_val > 1.0:
                            tariff_revenue_new[n] += (tau_val - 1.0) / tau_val * X_new[j][n] * pi_val_new
            income_new = {n: wage_income_new[n] + tariff_revenue_new[n] + deficits_map.get(country_list[n - 1], 0.0)
                          for n in range(1, N + 1)}
            X_new_upd = {j: {n: 0.0 for n in range(1, N + 1)} for j in range(1, J + 1)}
            for j in range(1, J + 1):
                for n in range(1, N + 1):
                    intermediate_demand = 0.0
                    for k in range(1, J + 1):
                        for i in range(1, N + 1):
                            gamma_inter = read_gama_1(j, k, n)
                            pi_val_new = read_pi(k, n, i)
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

        w_hat_new = {}
        for n in range(1, N + 1):
            numerator = 0.0
            for j in range(1, J + 1):
                for i in range(1, N + 1):
                    gamma_final = read_gama_2(j, n)
                    pi_val_new = read_pi(j, n, i)
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
        diff_outer = np.max(np.abs(np.array(list(w_hat_new.values())) - np.array(list(w_hat.values()))))
        for n in range(1, N + 1):
            w_hat[n] = (1 - epsilon) * w_hat[n] + epsilon * w_hat_new[n]
        print(f"Outer iteration {iter_outer}: max wage diff = {diff_outer:.8f}")
        if diff_outer < tol_outer:
            print(f"Convergence achieved after {iter_outer + 1} outer iterations.")
            break

        for n in range(1, N + 1):
            w_hat_arr[n - 1] = w_hat[n]

    w_hat_dict = {n: w_hat[n] for n in range(1, N + 1)}
    return w_hat_dict, X_new, {}  # pi_new omitted for brevity


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    X_initial, data, result_df = solve_initial_equilibrium_fixed_point()
    print("\nTask 1 completed. (First 5 sectors x 5 countries):")
    print(result_df.iloc[:5, :5])

    w_hat, X_new, pi_new = counterfactual_experiment(case=2)
    print("\nTask 2 completed. Counterfactual wage changes w_hat:")
    print(w_hat)
