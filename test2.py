from cgitb import reset
import pandas as pd
import numpy as np
from numpy.linalg import solve

# -------------------------------
# Step 0. Import Data Files
# -------------------------------
# Adjust file paths if your files are in a subfolder (e.g., 'data/')
alpha         = pd.read_csv('data/alpha.csv')         # Expected columns: 'country', 'sector', 'alpha'
countries_df  = pd.read_csv('data/CountryNames.csv')    # Expected columns: 'Country'
deficits      = pd.read_csv('data/Deficits.csv')        # Expected columns: 'country', 'Deficits'
gamma_io      = pd.read_csv('data/gamma_IO.csv')        # Expected columns: 'country', 'SectorOrigin', 'SectorDestination', 'gamma_IO'
gamma_va      = pd.read_csv('data/gamma_VA.csv')        # Expected columns: 'country', 'sector', 'gamma_VA'
one_plus_tau  = pd.read_csv('data/one_plus_tau.csv')    # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'one_plus_tau'
pi_df         = pd.read_csv('data/pi.csv')              # Expected columns: 'CountryOrigin', 'CountryDestination', 'sector', 'pi'
va_world      = pd.read_csv('data/VA_World.csv')

# Dimensions
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

print(one_plus_tao(40,31,31))
print(read_alpha(6,1))
print(read_pi(1,1,3))
print(read_gama_1(2,35,1))
print(read_gama_2(2,1))
print(read_d(5))