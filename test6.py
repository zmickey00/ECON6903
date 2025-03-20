import pandas as pd
import numpy as np
from numpy.linalg import inv
from joblib import Parallel, delayed

# =========================================
# Step 0. Import Data Files and Clean Data
# =========================================

alpha = pd.read_csv("alpha.csv")  # 'country', 'sector', 'alpha'
countries_df = pd.read_csv("CountryNames.csv")  # 'Country'
deficits = pd.read_csv("Deficits.csv")  # 'country', 'Deficits'
gamma_io = pd.read_csv("gamma_IO.csv")  # 'country','SectorOrigin','SectorDestination','gamma_IO'
gamma_va = pd.read_csv("gamma_VA.csv")  # 'country','sector','gamma_VA'
one_plus_tau = pd.read_csv("one_plus_tau.csv")  # 'CountryOrigin','CountryDestination','sector','one_plus_tau'
pi_df = pd.read_csv("pi.csv")  # 'CountryOrigin','CountryDestination','sector','pi'
va_world = pd.read_csv("VA_World.csv")

# Clean column names and strip extra spaces
for df in [alpha, countries_df, deficits, gamma_io, gamma_va, one_plus_tau, pi_df]:
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

# Normalize deficits using world value added
world_va = va_world['VA_World'].iloc[0]  # single value
deficits['Deficits'] = deficits['Deficits'] / world_va

# Dimensions
J = 40  # number of sectors
N = 31  # number of countries

# =========================================
# Create Lookup Dictionaries
# =========================================
alpha_dict = {(row['sector'], row['country']): row['alpha'] for _, row in alpha.iterrows()}
one_plus_tau_dict = {(row['sector'], row['CountryDestination'], row['CountryOrigin']): row['one_plus_tau']
                     for _, row in one_plus_tau.iterrows()}
pi_dict = {(row['sector'], row['CountryDestination'], row['CountryOrigin']): row['pi']
           for _, row in pi_df.iterrows()}
gamma_io_dict = {(row['SectorOrigin'], row['SectorDestination'], row['country']): row['gamma_IO']
                 for _, row in gamma_io.iterrows()}
gamma_va_dict = {(row['sector'], row['country']): row['gamma_VA']
                 for _, row in gamma_va.iterrows()}
deficits_dict = {row['country']: row['Deficits'] for _, row in deficits.iterrows()}

# =========================================
# Define the Required Lookup Functions
# (Kept exactly as in original)
# =========================================
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

# =========================================
# Pre-build global NumPy arrays for speed
# =========================================
alpha_arr = np.zeros((J, N))
for j_idx in range(J):
    for n_idx in range(N):
        alpha_arr[j_idx, n_idx] = read_alpha(j_idx+1, n_idx+1)

pi_arr = np.zeros((J, N, N))
for j_idx in range(J):
    for i_idx in range(N):
        for n_idx in range(N):
            pi_arr[j_idx, i_idx, n_idx] = read_pi(j_idx+1, i_idx+1, n_idx+1)

tau_arr = np.zeros((J, N, N))
for j_idx in range(J):
    for i_idx in range(N):
        for n_idx in range(N):
            tau_arr[j_idx, i_idx, n_idx] = one_plus_tao(j_idx+1, i_idx+1, n_idx+1)

gamma_io_arr = np.zeros((J, J, N))
for j_idx in range(J):
    for k_idx in range(J):
        for n_idx in range(N):
            gamma_io_arr[j_idx, k_idx, n_idx] = read_gama_1(j_idx+1, k_idx+1, n_idx+1)

gamma_va_arr = np.zeros((J, N))
for j_idx in range(J):
    for n_idx in range(N):
        gamma_va_arr[j_idx, n_idx] = read_gama_2(j_idx+1, n_idx+1)

deficits_arr = np.zeros(N)
for n_idx in range(N):
    deficits_arr[n_idx] = read_d(n_idx+1)

# =========================================
# Task 1: Accelerated Construction of M,B
# =========================================

def build_system_matrix_and_vector(J, N):
    """
    Builds the (J*N+1) x (J*N) matrix M and the (J*N+1) x 1 vector B in a vectorized manner,
    optionally parallelizing the slow loops with joblib if needed.
    """

    # -----------------------------
    # 1. sum_last_arr[j, n, k]
    # We'll do it in parallel with joblib to split the (j,n) or (j,n,k) dimension if it is large.
    # But typically J=40, N=31 => J*N=1240 => not extremely large, but let's demonstrate.
    def compute_sum_last_for_jn(j_idx, n_idx):
        # sum_{index} (tau_arr[k_idx,n_idx,index]-1)* pi_arr[k_idx,n_idx,index]/ tau_arr[k_idx,n_idx,index]
        # We'll build one row for all k_idx in [0..J-1].
        row_values = np.zeros(J)
        alpha_jn = alpha_arr[j_idx, n_idx]
        for k_idx in range(J):
            numer = (tau_arr[k_idx, n_idx, :] - 1.0) * pi_arr[k_idx, n_idx, :]
            denom = tau_arr[k_idx, n_idx, :]
            partial_sum = np.sum(numer / denom)
            row_values[k_idx] = alpha_jn * partial_sum
        return (j_idx, n_idx, row_values)

    # We run parallel over all (j_idx, n_idx) pairs
    results_sum_last = Parallel(n_jobs=-1, prefer="processes")(
        delayed(compute_sum_last_for_jn)(j_idx, n_idx)
        for j_idx in range(J)
        for n_idx in range(N)
    )
    # Store into sum_last_arr (shape J x N x J)
    sum_last_arr = np.zeros((J, N, J))
    for (j_idx, n_idx, row_vals) in results_sum_last:
        sum_last_arr[j_idx, n_idx, :] = row_vals

    # -----------------------------
    # 2. item_3_4D = sum_last_arr[j, n, k] if n == i else 0
    # shape => (J,N,J,N)
    item_3_4D = np.zeros((J, N, J, N))
    n_idx_grid = np.arange(N).reshape(1, N, 1, 1)
    i_idx_grid = np.arange(N).reshape(1, 1, 1, N)
    mask_n_eq_i = (n_idx_grid == i_idx_grid)  # shape => (1,N,1,N)
    sum_last_expanded = sum_last_arr[:, :, :, np.newaxis]  # shape => (J,N,J,1)
    item_3_4D = sum_last_expanded * mask_n_eq_i

    # -----------------------------
    # 3. item_2_4D[j,n,k,i] = (pi_arr[k,i,n]/tau_arr[k,i,n]) * [ gamma_io_arr[j,k,n]+ alpha_arr[j,n]* gamma_va_arr[k,n] ]
    def compute_item_2_block(j_idx, n_idx):
        # Return item_2_4D[j_idx,n_idx,k,i] for k in [0..J-1], i in [0..N-1].
        item_2_block = np.zeros((J, N))
        a_jn = alpha_arr[j_idx, n_idx]
        for k_idx in range(J):
            for i_idx in range(N):
                top = pi_arr[k_idx, i_idx, n_idx]
                bot = tau_arr[k_idx, i_idx, n_idx]
                g_io = gamma_io_arr[j_idx, k_idx, n_idx]
                g_va = gamma_va_arr[k_idx, n_idx]
                val = (top / bot) * (g_io + a_jn*g_va)
                item_2_block[k_idx, i_idx] = val
        return (j_idx, n_idx, item_2_block)

    results_item_2 = Parallel(n_jobs=-1, prefer="processes")(
        delayed(compute_item_2_block)(j_idx, n_idx)
        for j_idx in range(J)
        for n_idx in range(N)
    )
    item_2_4D = np.zeros((J, N, J, N))
    for (j_idx, n_idx, block2) in results_item_2:
        item_2_4D[j_idx, n_idx, :, :] = block2

    # Flatten item_2_4D, item_3_4D into (J*N, J*N)
    item_2_2D = item_2_4D.reshape(J*N, J*N)
    item_3_2D = item_3_4D.reshape(J*N, J*N)
    eye_2D = np.eye(J*N)
    M_flat = eye_2D - item_2_2D - item_3_2D

    # -----------------------------
    # 4. Build B
    B_vec = np.zeros(J*N + 1)
    for j_idx in range(J):
        for n_idx in range(N):
            row = j_idx*N + n_idx
            B_vec[row] = alpha_arr[j_idx, n_idx] * deficits_arr[n_idx]

    # -----------------------------
    # 5. Normalization row
    M_expanded = np.vstack([M_flat, np.zeros((1, J*N))])
    for k_idx in range(J):
        for i_idx in range(N):
            col = k_idx*N + i_idx
            val = 0.0
            for n_idx in range(N):
                val += gamma_va_arr[k_idx, n_idx]*(pi_arr[k_idx, i_idx, n_idx]/tau_arr[k_idx, i_idx, n_idx])
            M_expanded[J*N, col] = val
    B_vec[J*N] = 1.0

    # -----------------------------
    # 6. Remove one redundant row
    M_reduced = np.delete(M_expanded, J*N-1, axis=0)
    B_reduced = np.delete(B_vec, J*N-1, axis=0)

    return M_reduced, B_reduced

def solve_initial_equilibrium_fixed_point():
    """
    Solve for initial equilibrium expenditures X_n^j by writing the system M X = B,
    using the accelerated M,B construction plus a simple matrix solve.
    """
    M_reduced, B_reduced = build_system_matrix_and_vector(J, N)
    X_vector = np.linalg.solve(M_reduced, B_reduced)

    # Convert to nested dict X[j][n]
    X = {}
    for j_idx in range(J):
        j_sector = j_idx+1
        X[j_sector] = {}
        for n_idx in range(N):
            n_country = n_idx+1
            X[j_sector][n_country] = X_vector[j_idx*N + n_idx]

    # Save DataFrame
    X_matrix = X_vector.reshape(J, N)
    country_names_list = countries_df['Country'].tolist()
    sector_names = [f"Sector_{j}" for j in range(1, J+1)]
    result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_names_list)
    result_df.to_csv('initial_equilibrium_expenditures.csv')
    print("Task 1: Initial equilibrium expenditures saved to 'initial_equilibrium_expenditures.csv'")
    return X, {
        'alpha': alpha,
        'CountryNames': countries_df,
        'deficits': deficits,
        'gamma_IO': gamma_io,
        'gamma_VA': gamma_va,
        'one_plus_tau': one_plus_tau,
        'pi': pi_df
    }, result_df

# =========================================
# Task 2: Counterfactual Experiments with Parallelization
# =========================================

THETA = 4

def counterfactual_experiment(case=1, epsilon=0.1, max_iter_outer=200, tol_outer=1e-8):
    """
    Multi-processed + Vectorized solution for the counterfactual equilibrium.
    """

    print(f"\nExecuting counterfactual experiment Case {case}...")
    theta = THETA

    # 1. Load initial X and data
    X_init_nested, data_dict, _ = solve_initial_equilibrium_fixed_point()

    # Convert nested to array
    X_init_arr = np.zeros((J, N))
    for j_idx in range(J):
        for n_idx in range(N):
            X_init_arr[j_idx, n_idx] = X_init_nested[j_idx+1][n_idx+1]

    # 2. Build baseline lambda_hat, kappa_hat, deficits_new
    lambda_hat_arr = np.ones((J, N))
    kappa_hat_arr  = np.ones((J, N, N))
    deficits_new_arr = deficits_arr.copy()

    # Adjust by case
    if case == 2:
        deficits_new_arr[:] = 0.0
    elif case == 3:
        deficits_new_arr[:] = 0.0
        kappa_hat_arr[:] = 30.0
        # set diagonal to 1
        for j_idx in range(J):
            np.fill_diagonal(kappa_hat_arr[j_idx], 1.0)
    elif case == 4:
        # "China" index=7 => 1-based => 6 => 0-based
        china_idx = 6
        lambda_hat_arr[:, china_idx] = 5.0

    # 3. w_hat = 1
    w_hat_arr = np.ones(N)

    # Helper function for partial parallelization
    def compute_c_hat_for_jn(j_idx, n_idx, w_hat_local, P_hat_local):
        """
        Return c_hat for a single (j,n).
        c_hat[j,n] = w_hat[n]^gamma_va[j,n] * Π_k [P_hat[k,n]^gamma_io[j,k,n]].
        """
        gamma_final = gamma_va_arr[j_idx, n_idx]
        val = (w_hat_local[n_idx]**gamma_final)
        # multiply P_hat[k_idx,n_idx]^ gamma_io_arr[j_idx,k_idx,n_idx]
        # We'll just do a small loop here for clarity:
        for k_idx in range(J):
            exponent = gamma_io_arr[j_idx, k_idx, n_idx]
            if exponent != 0.0:
                val *= (P_hat_local[k_idx, n_idx]**exponent)
        return (j_idx, n_idx, val)

    def compute_P_hat_for_jn(j_idx, n_idx, c_hat_local):
        """
        Return updated P_hat[j,n].
        P_hat[j,n] = [ sum_i pi_arr[j,n,i] * lambda_hat_arr[j,i] * (c_hat[j,i]*kappa_hat_arr[j,n,i])^(-theta ) ]^(-1/theta).
        """
        summation = 0.0
        for i_idx in range(N):
            pi_val = pi_arr[j_idx, n_idx, i_idx]
            lam_val = lambda_hat_arr[j_idx, i_idx]
            c_kappa = c_hat_local[j_idx, i_idx] * kappa_hat_arr[j_idx, n_idx, i_idx]
            summation += pi_val * lam_val * (c_kappa**(-theta))
        if summation <= 0:
            return (j_idx, n_idx, 1.0)  # fallback
        val = summation**(-1.0/theta)
        return (j_idx, n_idx, val)

    # 4. Outer iteration for w_hat
    for iter_outer in range(max_iter_outer):

        # a) Inner iteration for P_hat
        P_hat_arr = np.ones((J, N))
        max_iter_inner = 100
        tol_inner = 1e-6

        for _ in range(max_iter_inner):
            # 1) Compute c_hat in parallel for each (j,n)
            #    c_hat[j,n] = w_hat[n]^gammaVA * ∏_{k} P_hat[k,n]^ gammaIO
            c_hat_temp = Parallel(n_jobs=-1, prefer="processes")(
                delayed(compute_c_hat_for_jn)(j_idx, n_idx, w_hat_arr, P_hat_arr)
                for j_idx in range(J)
                for n_idx in range(N)
            )
            c_hat_arr = np.zeros((J, N))
            for (j_idx, n_idx, val) in c_hat_temp:
                c_hat_arr[j_idx, n_idx] = val

            # 2) Compute new P_hat in parallel
            P_hat_temp = Parallel(n_jobs=-1, prefer="processes")(
                delayed(compute_P_hat_for_jn)(j_idx, n_idx, c_hat_arr)
                for j_idx in range(J)
                for n_idx in range(N)
            )
            P_hat_new = np.zeros((J, N))
            for (j_idx, n_idx, val) in P_hat_temp:
                P_hat_new[j_idx, n_idx] = val

            diff_inner = np.max(np.abs(P_hat_new - P_hat_arr))
            P_hat_arr = P_hat_new
            if diff_inner < tol_inner:
                break

        # b) Compute pi_new in parallel or vectorized
        #    pi_new[j,n,i] = pi_arr[j,n,i]* [ lambda_hat_arr[j,i] * ( (c_hat[j,i]*kappa_hat_arr[j,n,i])/P_hat[j,n] )^(-theta ) ]
        pi_new = np.zeros((J, N, N))

        def compute_pi_new_for_jni(j_idx, n_idx, i_idx):
            denom = P_hat_arr[j_idx, n_idx]
            ratio = (c_hat_arr[j_idx, i_idx]* kappa_hat_arr[j_idx, n_idx, i_idx]) / denom
            pi_hat = lambda_hat_arr[j_idx, i_idx] * (ratio**(-theta))
            return (j_idx, n_idx, i_idx, pi_arr[j_idx, n_idx, i_idx]*pi_hat)

        results_pi_new = Parallel(n_jobs=-1, prefer="processes")(
            delayed(compute_pi_new_for_jni)(j_idx, n_idx, i_idx)
            for j_idx in range(J)
            for n_idx in range(N)
            for i_idx in range(N)
        )
        for (j_idx, n_idx, i_idx, val) in results_pi_new:
            pi_new[j_idx, n_idx, i_idx] = val

        # c) Solve for X_new (income/spending iteration)
        X_new_arr = X_init_arr.copy()
        max_iter_income = 100
        tol_income = 1e-8

        def compute_wage_and_tariff_for_n(n_idx):
            """
            Return (wage_income_n, tariff_n) for a given n_idx
            by summing over j and i.
            """
            wage_val = 0.0
            tariff_val = 0.0
            for j_idx in range(J):
                # wage part => sum_{i} gammaVA * X_new[j,i]* pi_new[j,i,n]/tau
                for i_idx in range(N):
                    gamma_va_jn = gamma_va_arr[j_idx, n_idx]
                    tau_val     = tau_arr[j_idx, i_idx, n_idx]
                    wage_val   += gamma_va_jn * X_new_arr[j_idx, i_idx] * (pi_new[j_idx, i_idx, n_idx]/ tau_val)
                # tariff part => sum_{i} if tau>1 => (tau-1)/tau * X_new[j,n]* pi_new[j,n,i]
                tau_slice = tau_arr[j_idx, n_idx, :]
                pi_slice  = pi_new[j_idx, n_idx, :]
                x_jn      = X_new_arr[j_idx, n_idx]
                for i_idx in range(N):
                    tval = tau_slice[i_idx]
                    if tval>1.0:
                        tariff_val += ((tval-1.0)/tval) * x_jn* pi_slice[i_idx]

            return (n_idx, wage_val, tariff_val)

        for _ in range(max_iter_income):
            # 1) wage_income_new, tariff_revenue_new in parallel
            results_wage_tariff = Parallel(n_jobs=-1, prefer="processes")(
                delayed(compute_wage_and_tariff_for_n)(n_idx)
                for n_idx in range(N)
            )
            wage_income_new = np.zeros(N)
            tariff_revenue_new = np.zeros(N)
            for (n_idx, wv, tv) in results_wage_tariff:
                wage_income_new[n_idx] = wv
                tariff_revenue_new[n_idx] = tv

            income_new = wage_income_new + tariff_revenue_new + deficits_new_arr

            # 2) Update X_new_upd
            #    X_new_upd[j,n] = sum_{k,i} gamma_io[j,k,n]* X_new[k,i]* pi_new[k,i,n]/tau_arr[k,i,n] + alpha[j,n]*income_new[n]
            def compute_X_new_for_jn(j_idx, n_idx):
                intermed = 0.0
                for k_idx in range(J):
                    for i_idx in range(N):
                        gamma_io_val = gamma_io_arr[j_idx, k_idx, n_idx]
                        tau_val      = tau_arr[k_idx, i_idx, n_idx]
                        intermed    += gamma_io_val * X_new_arr[k_idx, i_idx] * (pi_new[k_idx, i_idx, n_idx]/ tau_val)
                final_demand = alpha_arr[j_idx, n_idx]* income_new[n_idx]
                return (j_idx, n_idx, intermed + final_demand)

            results_X_new = Parallel(n_jobs=-1, prefer="processes")(
                delayed(compute_X_new_for_jn)(j_idx, n_idx)
                for j_idx in range(J)
                for n_idx in range(N)
            )
            X_new_upd = np.zeros((J, N))
            for (j_idx, n_idx, val) in results_X_new:
                X_new_upd[j_idx, n_idx] = val

            total_va = np.sum(wage_income_new)
            if total_va>0:
                X_new_upd *= (1.0 / total_va)

            diff_income = np.max(np.abs(X_new_upd - X_new_arr))
            X_new_arr = X_new_upd
            if diff_income < tol_income:
                break

        # d) Update w_hat
        # w_hat_new[n] = numerator / old_income, with damping
        w_hat_new = np.zeros(N)
        for n_idx in range(N):
            numerator = 0.0
            old_val   = 0.0
            for j_idx in range(J):
                gamma_va_jn = gamma_va_arr[j_idx, n_idx]
                for i_idx in range(N):
                    tau_val = tau_arr[j_idx, i_idx, n_idx]
                    numerator += gamma_va_jn * X_new_arr[j_idx, i_idx] * (pi_new[j_idx, i_idx, n_idx]/tau_val)
                    # old
                    pi_old_val = pi_arr[j_idx, i_idx, n_idx]
                    old_val   += gamma_va_jn * X_init_arr[j_idx, i_idx]*( pi_old_val / tau_val )
            if old_val>0:
                w_hat_new[n_idx] = numerator/old_val
            else:
                w_hat_new[n_idx] = w_hat_arr[n_idx]

        diff_outer = np.max(np.abs(w_hat_new - w_hat_arr))
        w_hat_arr = (1.0 - epsilon)* w_hat_arr + epsilon* w_hat_new

        print(f"Outer iteration {iter_outer}: max wage diff = {diff_outer:.8e}")
        if diff_outer < tol_outer:
            print(f"Convergence achieved after {iter_outer + 1} outer iterations.")
            break

    # Convert final arrays to dictionary
    w_hat_dict = {n+1: w_hat_arr[n] for n in range(N)}
    X_new_dict = {}
    for j_idx in range(J):
        j_sec = j_idx+1
        X_new_dict[j_sec] = {}
        for n_idx in range(N):
            X_new_dict[j_sec][n_idx+1] = X_new_arr[j_idx, n_idx]

    # pi_new is (J,N,N); convert to nested
    pi_new_dict = {}
    for j_idx in range(J):
        j_sec = j_idx+1
        pi_new_dict[j_sec] = {}
        for n_idx in range(N):
            pi_new_dict[j_sec][n_idx+1] = {}
            for i_idx in range(N):
                pi_new_dict[j_sec][n_idx+1][i_idx+1] = pi_new[j_idx, n_idx, i_idx]

    return w_hat_dict, X_new_dict, pi_new_dict

# =========================================
# Main
# =========================================
if __name__ == "__main__":
    # Task 1
    X_initial, data, result_df = solve_initial_equilibrium_fixed_point()
    print("\n[INFO] Task 1 completed. (First 5 sectors x 5 countries):")
    print(result_df.iloc[:5, :5])

    # Task 2: e.g. case=2
    w_hat, X_new, pi_new = counterfactual_experiment(case=2)
    print("\n[INFO] Task 2 completed. Counterfactual wage changes (sample):")
    for c_idx in sorted(w_hat.keys())[:5]:
        print(f"  Country {c_idx}, w_hat={w_hat[c_idx]:.5f}")