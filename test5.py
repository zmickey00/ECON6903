import pandas as pd
import numpy as np
from numpy.linalg import inv

# =========================================
# Step 0. Import Data Files and Clean Data
# =========================================

alpha = pd.read_csv("alpha.csv")  # 'country', 'sector', 'alpha'
countries_df = pd.read_csv("CountryNames.csv")  # 'Country'
deficits = pd.read_csv("Deficits.csv")  # 'country', 'Deficits'
gamma_io = pd.read_csv("gamma_IO.csv")  # 'country', 'SectorOrigin', 'SectorDestination', 'gamma_IO'
gamma_va = pd.read_csv("gamma_VA.csv")  # 'country', 'sector', 'gamma_VA'
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
# (Kept exactly as in original request)
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
# Vector-Precomputation of Key Arrays
# =========================================
# We will build large NumPy arrays so that we can vectorize the operations.
# Indices (0-based) will map to: sector j in [0..J-1], country in [0..N-1].

# For alpha (shape J x N), alpha_arr[j, n] = alpha of sector j+1, country n+1
alpha_arr = np.zeros((J, N))
for j_idx in range(J):
    for n_idx in range(N):
        alpha_arr[j_idx, n_idx] = read_alpha(j_idx+1, n_idx+1)

# For pi (shape J x N x N). We'll interpret:
#   j => sector, i => CountryDestination, origin => CountryOrigin
#   so pi_arr[j, i, origin]
pi_arr = np.zeros((J, N, N))
for j_idx in range(J):
    for i_idx in range(N):
        for n_idx in range(N):
            pi_arr[j_idx, i_idx, n_idx] = read_pi(j_idx+1, i_idx+1, n_idx+1)

# For tau (shape J x N x N), same indexing as pi_arr
tau_arr = np.zeros((J, N, N))
for j_idx in range(J):
    for i_idx in range(N):
        for n_idx in range(N):
            tau_arr[j_idx, i_idx, n_idx] = one_plus_tao(j_idx+1, i_idx+1, n_idx+1)

# For gamma_io (shape: J x J x N). gamma_io_arr[j, k, n] = gamma_IO(sectorOrigin=j+1, sectorDestination=k+1, country=n+1)
gamma_io_arr = np.zeros((J, J, N))
for j_idx in range(J):
    for k_idx in range(J):
        for n_idx in range(N):
            gamma_io_arr[j_idx, k_idx, n_idx] = read_gama_1(j_idx+1, k_idx+1, n_idx+1)

# For gamma_va (shape: J x N). gamma_va_arr[j, n] = gamma_VA(sector=j+1, country=n+1)
gamma_va_arr = np.zeros((J, N))
for j_idx in range(J):
    for n_idx in range(N):
        gamma_va_arr[j_idx, n_idx] = read_gama_2(j_idx+1, n_idx+1)

# For deficits (shape: N, i.e. one value per country)
deficits_arr = np.zeros(N)
for n_idx in range(N):
    deficits_arr[n_idx] = read_d(n_idx+1)

# =========================================
# Task 1: Solve for the Initial Equilibrium (Accelerated)
# =========================================

def build_system_matrix_and_vector(J, N):
    """
    Builds the (J*N+1) x (J*N) matrix M and the (J*N+1) x 1 vector B in a vectorized manner.
    Uses the pre-built arrays: alpha_arr, pi_arr, tau_arr, gamma_io_arr, gamma_va_arr, deficits_arr.
    """

    # -------------------------------
    # 1. Precompute sum_last for item_3
    # sum_last[j, n, k] = alpha(j,n) * sum_{index}( (tau(k,n,index)-1)*pi(k,n,index)/tau(k,n,index) )
    # where sector k => k in [0..J-1], country destination n => n in [0..N-1], origin => index in [0..N-1].
    # We'll store as sum_last_arr shape (J, N, J).
    sum_last_arr = np.zeros((J, N, J))
    for j_idx in range(J):
        for n_idx in range(N):
            # For each k, do the sum over "index"
            for k_idx in range(J):
                numer = (tau_arr[k_idx, n_idx, :] - 1.0) * pi_arr[k_idx, n_idx, :]
                denom = tau_arr[k_idx, n_idx, :]
                sum_over_index = np.sum(numer / denom)  # sum over origin dimension
                sum_last_arr[j_idx, n_idx, k_idx] = alpha_arr[j_idx, n_idx] * sum_over_index

    # -------------------------------
    # 2. item_3_4D[j, n, k, i] = sum_last_arr[j, n, k] if n == i else 0
    # We'll create a 4D array (J, N, J, N) so that row=(j,n), col=(k,i).
    # Then later we flatten to (J*N, J*N).
    item_3_4D = np.zeros((J, N, J, N))
    # "n == i" can be done via a mask:
    n_idx_grid = np.arange(N).reshape(1, N, 1, 1)  # shape (1,N,1,1)
    i_idx_grid = np.arange(N).reshape(1, 1, 1, N)  # shape (1,1,1,N)
    mask_n_eq_i = (n_idx_grid == i_idx_grid)       # shape (1,N,1,N) of booleans

    # We want sum_last_arr[j, n, k] broadcast into the i dimension if n==i
    # sum_last_arr shape is (J,N,J). Expand a dimension for i:
    sum_last_expanded = sum_last_arr[:, :, :, np.newaxis]  # shape (J,N,J,1)
    item_3_4D = sum_last_expanded * mask_n_eq_i  # shape (J,N,J,N)

    # -------------------------------
    # 3. item_2_4D[j, n, k, i] = (pi(k, i, n)/tau(k, i, n)) * [ gamma_io(j, k, n) + alpha(j, n)*gamma_va(k, n) ]
    # Also shape (J,N,J,N).
    item_2_4D = np.zeros((J, N, J, N))
    for j_idx in range(J):
        for n_idx in range(N):
            a_jn = alpha_arr[j_idx, n_idx]
            for k_idx in range(J):
                for i_idx in range(N):
                    top = pi_arr[k_idx, i_idx, n_idx]
                    bot = tau_arr[k_idx, i_idx, n_idx]
                    g_io = gamma_io_arr[j_idx, k_idx, n_idx]
                    g_va = gamma_va_arr[k_idx, n_idx]
                    val = (top / bot) * (g_io + a_jn * g_va)
                    item_2_4D[j_idx, n_idx, k_idx, i_idx] = val

    # -------------------------------
    # 4. Combine item_1, item_2, item_3 into M
    # Flatten item_2_4D, item_3_4D from shape (J,N,J,N) => (J*N, J*N).
    item_2_2D = item_2_4D.reshape(J*N, J*N)
    item_3_2D = item_3_4D.reshape(J*N, J*N)
    eye_2D    = np.eye(J*N)
    M_flat    = eye_2D - item_2_2D - item_3_2D  # shape (J*N, J*N)

    # -------------------------------
    # 5. Build B (size J*N+1). For row j*N + n => alpha(j,n)*Deficits(n)
    B_vec = np.zeros(J*N + 1)
    for j_idx in range(J):
        for n_idx in range(N):
            row = j_idx*N + n_idx
            B_vec[row] = alpha_arr[j_idx, n_idx] * deficits_arr[n_idx]

    # -------------------------------
    # 6. Add one extra row for normalization condition:
    # M[J*N, col=(k*N + i)] = sum_{n} gamma_va_arr[k, n] * pi_arr[k, i, n]/tau_arr[k, i, n]
    # Then B[J*N] = 1
    M_expanded = np.vstack([M_flat, np.zeros((1, J*N))])  # shape => (J*N+1, J*N)
    for k_idx in range(J):
        for i_idx in range(N):
            col = k_idx*N + i_idx
            val = 0.0
            for n_idx in range(N):
                val += gamma_va_arr[k_idx, n_idx] * (pi_arr[k_idx, i_idx, n_idx]/tau_arr[k_idx, i_idx, n_idx])
            M_expanded[J*N, col] = val
    B_vec[J*N] = 1.0

    # -------------------------------
    # 7. Remove one redundant row: row index=J*N-1
    M_reduced = np.delete(M_expanded, J*N - 1, axis=0)
    B_reduced = np.delete(B_vec, J*N - 1, axis=0)

    return M_reduced, B_reduced

def solve_initial_equilibrium_fixed_point():
    """
    Solve for initial equilibrium expenditures X_n^j by writing the system in long form and solving M X = B.
    """
    # 1. Build M and B
    M_reduced, B_reduced = build_system_matrix_and_vector(J, N)  # shape => (J*N, J*N), (J*N,)

    # 2. Solve
    X_vector = np.linalg.solve(M_reduced, B_reduced)  # shape => (J*N,)

    # 3. Convert X_vector -> X[j][n]
    X = {}
    for j_idx in range(1, J+1):
        X[j_idx] = {}
        for n_idx in range(1, N+1):
            index = (j_idx - 1)*N + (n_idx - 1)
            X[j_idx][n_idx] = X_vector[index]

    # 4. Save DataFrame
    X_matrix = X_vector.reshape(J, N)  # shape => (J,N)
    country_names_list = countries_df['Country'].tolist()
    sector_names = [f"Sector_{j}" for j in range(1, J+1)]
    result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_names_list)
    result_df.to_csv('initial_equilibrium_expenditures.csv', index=True)
    print("Task 1: Initial equilibrium expenditures saved to 'initial_equilibrium_expenditures.csv'")

    # Return
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
# Task 2: Counterfactual Experiments
# Accelerated version
# =========================================

THETA = 4

def counterfactual_experiment(case=1, epsilon=0.1, max_iter_outer=200, tol_outer=1e-8):
    """
    Solve for the counterfactual equilibrium in an accelerated manner.
    case in {1,2,3,4}.
    """

    print(f"\nExecuting counterfactual experiment Case {case}...")
    theta = THETA

    # 1. Load initial X and data (runs Task 1 in the process).
    X_initial_nested, data_dict, _ = solve_initial_equilibrium_fixed_point()

    # Convert X_initial_nested -> a NumPy array (J x N) for faster array usage:
    X_initial_arr = np.zeros((J, N))
    for j_idx in range(J):
        for n_idx in range(N):
            X_initial_arr[j_idx, n_idx] = X_initial_nested[j_idx+1][n_idx+1]

    # 2. Build baseline lambda_hat, kappa_hat, deficits_new (vector form).
    #    lambda_hat: shape (J, N), all ones
    #    kappa_hat: shape (J, N, N), all ones
    lambda_hat_arr = np.ones((J, N))
    kappa_hat_arr  = np.ones((J, N, N))
    deficits_new_arr = deficits_arr.copy()  # shape (N,)

    # Case adjustments:
    # 2 => D'_n=0
    # 3 => D'_n=0 + autarky approx => kappa_hat=30 if n!=i
    # 4 => Productivity shock in China (country index=7 => 1-based => 6 in 0-based)
    if case == 2:
        deficits_new_arr[:] = 0.0
    elif case == 3:
        deficits_new_arr[:] = 0.0
        # kappa_hat[j, n, i] = 30 if n != i
        for j_idx in range(J):
            # fill all 30
            kappa_hat_arr[j_idx, :, :] = 30.0
            # diagonal = 1
            np.fill_diagonal(kappa_hat_arr[j_idx, :, :], 1.0)
    elif case == 4:
        #  "China" = index 7 => 1-based => (7-1)=6 in 0-based
        china_idx = 6
        lambda_hat_arr[:, china_idx] = 5.0

    # 3. Initialize w_hat = 1 for all countries => shape (N,)
    w_hat_arr = np.ones(N)

    # 4. Outer iteration: update w_hat until convergence
    for iter_outer in range(max_iter_outer):

        # ----------------------------------
        # (a) Inner iteration to solve for P_hat
        #     We'll do a simple fixed-point iteration on P_hat.
        #     P_hat shape = (J, N)
        #     Start from P_hat=1. We'll iterate up to some small tolerance.
        P_hat_arr = np.ones((J, N))
        max_iter_inner = 100
        tol_inner = 1e-6

        for _ in range(max_iter_inner):
            # c_hat[j,n] = w_hat_arr[n]^gamma_va_arr[j,n] * ∏_{k} [P_hat_arr[k,n]^gamma_io_arr[j,k,n]]
            # We'll do a log-based approach or direct multiplication for speed.
            # Direct multiplication:
            c_hat = np.zeros((J, N))
            for j_idx in range(J):
                for n_idx in range(N):
                    # base = w_hat_arr[n_idx]^gamma_va
                    base_val = w_hat_arr[n_idx]**(gamma_va_arr[j_idx, n_idx])
                    # multiply by each sector k's P_hat^gamma_io
                    # gamma_io_arr[j_idx, k_idx, n_idx], k_idx in [0..J-1]
                    # P_hat_arr[k_idx, n_idx]^ gamma_io
                    # We can do a product of the form np.prod( P_hat_arr[:, n_idx]**(gamma_io_arr[j_idx, :, n_idx]) )
                    # or we do exponent-sum of logs. We'll do direct for clarity:
                    power_arr = P_hat_arr[:, n_idx]**(gamma_io_arr[j_idx, :, n_idx])
                    c_hat[j_idx, n_idx] = base_val * np.prod(power_arr)

            # P_hat_new[j,n] = { sum_{i} [ pi_arr[j, n, i]* lambda_hat_arr[j,i] * ( c_hat[j,i]*kappa_hat_arr[j,n,i] )^(-theta ) ] }^(-1/theta)
            # We sum over i => axis=?
            # But we have j,n fixed. So let's do a 2D pass:
            P_hat_new = np.zeros((J, N))
            for j_idx in range(J):
                for n_idx in range(N):
                    # shape (N,) over i => i in [0..N-1]
                    # pi_arr[j_idx, n_idx, i] * lambda_hat_arr[j_idx, i] * [ c_hat[j_idx, i]*kappa_hat_arr[j_idx, n_idx, i] ]^(-theta)
                    pi_slice   = pi_arr[j_idx, n_idx, :]
                    lam_slice  = lambda_hat_arr[j_idx, :]
                    c_kappa    = c_hat[j_idx, :] * kappa_hat_arr[j_idx, n_idx, :]
                    # raise c_kappa to (-theta)
                    sum_term   = np.sum( pi_slice * lam_slice * (c_kappa**(-theta)) )
                    if sum_term > 0.0:
                        P_hat_new[j_idx, n_idx] = sum_term**(-1.0/theta)
                    else:
                        P_hat_new[j_idx, n_idx] = P_hat_arr[j_idx, n_idx]

            # check convergence
            diff_inner = np.max(np.abs(P_hat_new - P_hat_arr))
            P_hat_arr = P_hat_new
            if diff_inner < tol_inner:
                break

        # ----------------------------------
        # (b) Compute pi_new = pi_arr * pi_hat, where
        #     pi_hat[j,n,i] = lambda_hat_arr[j,i] * [ (c_hat[j,i]*kappa_hat_arr[j,n,i]) / P_hat_arr[j,n] ]^(-theta)
        pi_new = np.zeros((J, N, N))
        for j_idx in range(J):
            for n_idx in range(N):
                denom = P_hat_arr[j_idx, n_idx]
                for i_idx in range(N):
                    ratio = (c_hat[j_idx, i_idx] * kappa_hat_arr[j_idx, n_idx, i_idx]) / denom
                    pi_hat_val = lambda_hat_arr[j_idx, i_idx] * (ratio**(-theta))
                    pi_new[j_idx, n_idx, i_idx] = pi_arr[j_idx, n_idx, i_idx] * pi_hat_val

        # ----------------------------------
        # (c) Solve for new X_new with an income/spending iteration
        #     We'll do up to ~100 loops or until stable.
        X_new = X_initial_arr.copy()  # shape (J,N), start from initial or last iteration
        max_iter_income = 100
        tol_income = 1e-8

        for _ in range(max_iter_income):
            # wage_income_new[n] = sum_{j,i} gamma_va_arr[j,n]* X_new[j,i]* pi_new[j,i,n]/tau_arr[j,i,n]
            # tariff_revenue_new[n] = sum_{j,i} if tau>1 => (tau-1)/tau * X_new[j,n]* pi_new[j,n,i]
            wage_income_new = np.zeros(N)
            tariff_revenue_new = np.zeros(N)

            # We'll do partial vectorization:
            # wage_income_new[n] => sum_{j,i}
            for n_idx in range(N):
                # 1) wage_income_new
                #   sum over j,i: gamma_va_arr[j,n]* X_new[j,i]* pi_new[j,i,n]/ tau_arr[j,i,n]
                #   but note that in pi_new, j is sector, the second index is n, the third is i => pi_new[j, n, i].
                #   We'll reorder the usage to match the formula in code:
                #   Actually the code does: pi_new[k, i, n]? There's an index mismatch risk. We keep the code's logic:
                #   In this final version, we just replicate the original formula carefully:
                for j_idx in range(J):
                    gamma_va_jn = gamma_va_arr[j_idx, n_idx]
                    for i_idx in range(N):
                        # X_new[j_idx, i_idx], pi_new => j_idx, i_idx => ??? We have j_idx in pi_new? Possibly reversed.
                        # The original code used read_pi(k,i,n)? We used pi_arr[k, i, n], so we keep that:
                        # But now pi_new is shape (J,N,N) => pi_new[j, i, n]? Actually we set it as pi_new[j, n, i].
                        # Let's adapt the formula: wage_income_new[n] += gamma_va_jn * X_new[j_idx, i_idx] * pi_new[j_idx, i_idx, n_idx]/ tau. That means we might want to define an alternative ordering.
                        # To keep consistent with your M-building code, let's do it directly:

                        pi_val_new = pi_new[j_idx, i_idx, n_idx]  # j => j_idx, destination => i_idx, origin => n_idx
                        tau_val    = tau_arr[j_idx, i_idx, n_idx]
                        wage_income_new[n_idx] += gamma_va_jn * X_new[j_idx, i_idx] * (pi_val_new / tau_val)

                # 2) tariff_revenue_new[n]
                #    sum_{j,i} if tau>1 => (tau-1)/tau * X_new[j,n]* pi_new[j,n,i]
                #    Interpreting carefully: j => sector, n => destination, i => origin => pi_new[j,n,i]
                for j_idx in range(J):
                    for i_idx in range(N):
                        tau_val = tau_arr[j_idx, n_idx, i_idx]
                        pi_val_new = pi_new[j_idx, n_idx, i_idx]
                        if tau_val > 1.0:
                            tariff_revenue_new[n_idx] += ((tau_val - 1.0)/tau_val) * X_new[j_idx, n_idx] * pi_val_new

            income_new = wage_income_new + tariff_revenue_new + deficits_new_arr

            # 3) X_new_upd[j,n] = intermediate_demand + final_demand, scaled by world VA
            #    intermediate = sum_{k,i} gamma_io_arr[j,k,n]* X_new[k,i]* pi_new[k,i,n]/ tau_arr[k,i,n]
            #    final = alpha_arr[j,n]* income_new[n]
            X_new_upd = np.zeros((J, N))
            for j_idx in range(J):
                for n_idx in range(N):
                    # intermediate
                    intermed = 0.0
                    for k_idx in range(J):
                        for i_idx in range(N):
                            gamma_io_val = gamma_io_arr[j_idx, k_idx, n_idx]
                            pi_val_new   = pi_new[k_idx, i_idx, n_idx]
                            tau_val      = tau_arr[k_idx, i_idx, n_idx]
                            intermed    += gamma_io_val * X_new[k_idx, i_idx] * (pi_val_new / tau_val)
                    final_demand = alpha_arr[j_idx, n_idx]* income_new[n_idx]
                    X_new_upd[j_idx, n_idx] = intermed + final_demand

            total_va = np.sum(wage_income_new)
            if total_va > 0:
                scale = 1.0 / total_va
                X_new_upd *= scale

            diff_income = np.max(np.abs(X_new_upd - X_new))
            X_new = X_new_upd
            if diff_income < tol_income:
                break

        # ----------------------------------
        # (d) Update w_hat:
        # w_hat_new[n] = [ sum_{j,i} gamma_va_arr[j,n]* X_new[j,i]* pi_new[j,i,n]/tau_arr[j,i,n ] ] / [ sum_{j,i} gamma_va_arr[j,n]* X_initial_arr[j,i]* pi_arr[j,i,n]/tau_arr[j,i,n ] ]
        w_hat_new = np.zeros(N)
        denom_old = np.zeros(N)  # old income
        for n_idx in range(N):
            # numerator
            numerator = 0.0
            old_val   = 0.0
            for j_idx in range(J):
                gamma_va_jn = gamma_va_arr[j_idx, n_idx]
                for i_idx in range(N):
                    tau_val = tau_arr[j_idx, i_idx, n_idx]
                    numerator += gamma_va_jn * X_new[j_idx, i_idx] * pi_new[j_idx, i_idx, n_idx] / tau_val
                    # old
                    pi_val_old = pi_arr[j_idx, i_idx, n_idx]
                    old_val   += gamma_va_jn * X_initial_arr[j_idx, i_idx] * pi_val_old / tau_val

            w_hat_new[n_idx] = numerator/old_val if old_val>0 else w_hat_arr[n_idx]

        diff_outer = np.max(np.abs(w_hat_new - w_hat_arr))
        # damp
        w_hat_arr = (1 - epsilon)*w_hat_arr + epsilon*w_hat_new

        print(f"Outer iteration {iter_outer}: max wage diff = {diff_outer:.8e}")
        if diff_outer < tol_outer:
            print(f"Convergence achieved after {iter_outer + 1} outer iterations.")
            break

    # Return final
    # Convert w_hat_arr to dictionary form if desired
    w_hat_dict = {n+1: w_hat_arr[n] for n in range(N)}

    # Convert X_new to nested dictionary
    X_new_dict = {}
    for j_idx in range(J):
        X_new_dict[j_idx+1] = {}
        for n_idx in range(N):
            X_new_dict[j_idx+1][n_idx+1] = X_new[j_idx, n_idx]

    # Convert pi_new to nested dictionary
    pi_new_dict = {}
    for j_idx in range(J):
        pi_new_dict[j_idx+1] = {}
        for n_idx in range(N):
            pi_new_dict[j_idx+1][n_idx+1] = {}
            for i_idx in range(N):
                pi_new_dict[j_idx+1][n_idx+1][i_idx+1] = pi_new[j_idx, n_idx, i_idx]

    return w_hat_dict, X_new_dict, pi_new_dict

# =========================================
# Main
# =========================================
if __name__ == "__main__":
    # ---- Task 1 ----
    X_initial, data, result_df = solve_initial_equilibrium_fixed_point()
    print("\n[INFO] Task 1 completed. (First 5 sectors x 5 countries):")
    print(result_df.iloc[:5, :5])

    # ---- Task 2 ----
    # Example: run the counterfactual with case=2
    w_hat, X_new, pi_new = counterfactual_experiment(case=2)
    print("\n[INFO] Task 2 completed. Counterfactual wage changes w_hat (sample):")
    sample_countries = list(w_hat.keys())[:5]
    for c in sample_countries:
        print(f" Country {c}: w_hat={w_hat[c]:.4f}")