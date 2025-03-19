
# %%
import pandas as pd
import numpy as np
from numpy.linalg import solve

# %%
# -------------------------------
# Step 0. Import Data Files
# -------------------------------
# Adjust file paths if your files are in a subfolder (e.g., 'data/')
#dataFolder = "C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise"
alpha         = pd.read_csv("alpha.csv")         # Expected columns: 'country', 'sector', 'alpha'
countries_df  = pd.read_csv("CountryNames.csv")    # Expected columns: 'Country'
deficits      = pd.read_csv("Deficits.csv")        # Expected columns: 'country', 'Deficits'
gamma_io      = pd.read_csv("gamma_IO.csv")        # Expected columns: 'country', 'SectorOrigin', 'SectorDestination', 'gamma_IO'
gamma_va      = pd.read_csv("gamma_VA.csv" )      # Expected columns: 'country', 'sector', 'gamma_VA'
one_plus_tau  = pd.read_csv("one_plus_tau.csv")    # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'one_plus_tau'
pi_df         = pd.read_csv("pi.csv")              # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'pi'
va_world      = pd.read_csv("VA_World.csv")
#country_names = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/CountryNames.csv")

# %%
print(va_world.columns)

# %%
#Normalization
world_va = va_world['VA_World'].iloc[0]   # Assuming a single value is provided
deficits['Deficits'] = deficits['Deficits'] / world_va

# %%

# Dimensions
J = 40  # number of sectors (or whichever grouping you have)
N = 31  # number of countries (or whichever grouping you have)

# %%
print(alpha.columns)


# %%
# For alpha, key: (sector, country)
#alpha_dict = { (row['sector'], row['country']): row['alpha'] for _, row in alpha.iterrows() }
alpha.columns = alpha.columns.str.strip()  # Remove extra spaces
alpha_dict = { (row['sector'], row['country']): row['alpha'] for _, row in alpha.iterrows() }


# For one_plus_tau, key: (sector, destination, origin)
one_plus_tau.columns = one_plus_tau.columns.str.strip()  # Remove extra spaces
one_plus_tau_dict = { (row['sector'], row['CountryDestination'], row['CountryOrigin']): row['one_plus_tau']
                      for _, row in one_plus_tau.iterrows() }

# For pi, key: (sector, destination, origin)
pi_df.columns = pi_df.columns.str.strip()  # Remove extra space
pi_dict = { (row['sector'], row['CountryDestination'], row['CountryOrigin']): row['pi']
            for _, row in pi_df.iterrows() }

# For gamma_io, key: (SectorOrigin, SectorDestination, country)
gamma_io.columns = gamma_io.columns.str.strip()  # Remove extra space
gamma_io_dict = { (row['SectorOrigin'], row['SectorDestination'], row['country']): row['gamma_IO']
                  for _, row in gamma_io.iterrows() }

# For gamma_va, key: (sector, country)
gamma_va.columns = gamma_va.columns.str.strip()  # Remove extra space
gamma_va_dict = { (row['sector'], row['country']): row['gamma_VA']
                  for _, row in gamma_va.iterrows() }

# For deficits, key: country
deficits.columns = deficits.columns.str.strip()  # Remove extra space
deficits_dict = { row['country']: row['Deficits'] for _, row in deficits.iterrows() }




# %%
# -------------------------------------
# Define lookup functions using dictionaries
# -------------------------------------

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


# %%
# -------------------------------------
# Other functions as in your original code
# -------------------------------------

def sum_last(j, n, k):
    total = 0
    # index represents the origin country (l)
    for index in range(1, N+1):  # assumes 1-based indexing for country indices
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

# -------------------------------------
# Preallocate and fill matrix M using 0-based loops
# -------------------------------------

# We'll use 0-based loops here and add 1 to the indices when calling m_entry.
M = np.zeros((J * N+1, J * N))

def fill_m():
    # Loop over sectors and countries in 0-based indexing.
    for j in range(J):         # j = 0,1,...,J-1; sector = j+1
        for n in range(N):     # n = 0,1,...,N-1; country = n+1
            for k in range(J): # k = 0,1,...,J-1; sector = k+1
                for i in range(N): # i = 0,1,...,N-1; country = i+1
                    m_row = j * N + n
                    m_col = k * N + i
                    # Adjust indices to 1-based when calling m_entry and related functions
                    M[m_row, m_col] = m_entry(n + 1, i + 1, j + 1, k + 1)
    return M

fill_m()
print(M)

#def b_entry(j,n):
#    pass



# print(read_pi(2, 3,  2))
# print(one_plus_tao(2, 3,  2))
# print(read_gama_1(1,2,2))
# print(read_gama_2(2,2))
# print(read_alpha(1,2))
#function b=fillb()
#    for n =1:N
#        for j =1:J
#            b_index = (j - 1) * N + n;
#            b(b_index)=alphaLookupNumeric(j,n)*deficitLookupNumeric(n);
#        end
#    end
#end
# Optionally, print or inspect a portion of M

#
# # Convert the NumPy array M to a DataFrame
# M_df = pd.DataFrame(M_filled)
#
# # Export the DataFrame to a CSV file
# M_df.to_csv('M.csv', index=False)


# %%
B = np.zeros((J * N+1, 1))
def product(j,n):
    return read_alpha(j, n)*read_d(n)
def fill_b():
    for j in range(J):
        for n in range(N):
            b_row = j * N + n
            B[b_row,0]=product(j+1,n+1)
    
    return B

fill_b()
print(B)




# %%
# print(M[J*N,3])

# %%
for k in range(J):
    for i in range(N):
        m_col = k * N + i
        for n in range(N):
            M[J*N,m_col]+=read_gama_2(k+1, n+1)* read_pi(k+1, i+1, n+1)/one_plus_tao(k+1, i+1, n+1)

print(M)
            

# %%
B[J*N,0]=1
print(B)

# %%
M_reduced=np.delete(M,1239,axis=0)
B_reduced=np.delete(B,1239,axis=0)

print(M_reduced.shape)
print(B_reduced.shape)


# %%
M_inv=np.linalg.inv(M_reduced)
print(M_inv)

# %%
X=np.dot(M_inv,B_reduced)
print(X)

print(X.shape)
print(X[1])

sum1 = 0

for i in range(1240):
    sum1 += X[i]

print(sum1)
# # %%
# print(pi_df.shape)
#
# # %%
# sum=0
# for i in range(40):
#     sum+= X[i,0]
#
#
# print(sum)
#
# # %%
# print(X[93,0])
# print(read_pi(1, 3, 2))
#
# # %%
# print(read_gama_1(1,2,3))
#
# # %%
# print(one_plus_tao(20, 12, 25))
#
# # %%
# X_country=np.zeros((31, 1))
# for i in range(31):
#     sum=0
#     for j in range(40):
#         sum+=X[31*i+j]
#     X_country[i]=sum
#
# print(X_country)
#
# for i in range (1240):
#     sum1 =0
#     sum1 += X_country[i]
#
# print(sum1)
# %%
# def load_data():
#     print("读取数据文件...")
#     alpha = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/alpha.csv")
#     country_names = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/CountryNames.csv")
#     gamma_VA = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/gamma_VA.csv" )
#     gamma_IO = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/gamma_IO.csv")
#     va_world = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/VA_World.csv")
#     pi = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/pi.csv")
#     one_plus_tau = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/one_plus_tau.csv")
#     deficits = pd.read_csv("C:/Users/zzk20/Others/Econ 690/3/Assignment_CP_2025 (1)/Assignment_CP_2025/Data_Exercise/Deficits.csv")
#
#     # 清理列名（移除可能的空格）
#     for df in [country_names, alpha, gamma_VA, gamma_IO, va_world, pi, one_plus_tau, deficits]:
#         df.columns = [col.strip() for col in df.columns]
#
#         # 同时清理数据列中的空格
#         for col in df.columns:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].str.strip()
#
#     return {
#         'country_names': country_names,
#         'alpha': alpha,
#         'gamma_VA': gamma_VA,
#         'gamma_IO': gamma_IO,
#         'va_world': va_world,
#         'pi': pi,
#         'one_plus_tau': one_plus_tau,
#         'deficits': deficits
#     }
#
# # %%
# data = load_data()
# country_names_df = data['country_names']
# N = len(country_names_df)  # 国家数量
# J = len(data['alpha']['sector'].unique())  # 行业部门数量
# print(f"模型维度: {N} 个国家, {J} 个行业部门")
#
# X_matrix=np.zeros((31,40))
#
# for i in range(31):
#     for j in range(40):
#         X_matrix[i,j]=X[31*i+j]
#
# print(X_matrix)
#
# #X_matrix_prime=X_matrix.T
#
# # 创建列名（使用国家名称）
# country_names_list = []
# for _, row in country_names_df.iterrows():
#     country_names_list.append(row['Country'])
#
# # 创建行名（部门编号）
# sector_names = [f"Sector_{j}" for j in range(1, J + 1)]
#
# # 创建DataFrame
# result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_names_list)
# result_df.to_csv('initial_equilibrium_expenditures.csv')
#
# X_matrix = np.zeros((J, N))
# for j in range(1, J + 1):
#     for n in range(1, N + 1):
#         X_matrix[j-1, n-1] = X[j][n]
#
# # 创建列名（使用国家名称）
# country_names_list = []
# for _, row in country_names_df.iterrows():
#     country_names_list.append(row['Country'])
#
# # 创建行名（部门编号）
# sector_names = [f"Sector_{j}" for j in range(1, J + 1)]
#
# # 创建DataFrame
# result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_names_list)
#
#
# # %%
# X_matrix = np.zeros((J, N))
# for j in range(40):
#     for n in range(31):
#         X_matrix[j, n] = X[31*(j)+n]
#
# # 创建列名（使用国家名称）
# country_names_list = []
# for _, row in country_names_df.iterrows():
#     country_names_list.append(row['Country'])
#
# # 创建行名（部门编号）
# sector_names = [f"Sector_{j}" for j in range(1, J + 1)]
#
# # 创建DataFrame
# result_df = pd.DataFrame(X_matrix, index=sector_names, columns=country_names_list)
#
# # 保存到CSV文件
# result_df.to_csv('initial_equilibrium_expenditures.csv')
# print("结果已保存至 'initial_equilibrium_expenditures.csv'")
#
#

# %%



