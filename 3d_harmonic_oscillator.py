import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh

# =========================
# Parameters
# =========================
N = 30  # grid points per dimension (keep small to avoid memory issues)
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Choose potential: 'harmonic' or 'double_well'
potential_type = 'double_well'

# =========================
# 1D Laplacian
# =========================
diag = -2.0 * np.ones(N)
off_diag = np.ones(N-1)
D = diags([off_diag, diag, off_diag], offsets=[-1,0,1]) / dx**2
I = eye(N)

# 3D Laplacian using Kronecker products
L = kron(kron(I,I),D) + kron(kron(I,D),I) + kron(kron(D,I),I)

# =========================
# 3D Potential
# =========================
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

if potential_type == 'harmonic':
    V = 0.5*(X**2 + Y**2 + Z**2)
elif potential_type == 'double_well':
    V = X**4 + Y**4 + Z**4 - 5*(X**2 + Y**2 + Z**2)
else:
    raise ValueError("Unknown potential type.")

V_flat = V.flatten()
V_matrix = diags(V_flat, 0)

# =========================
# Hamiltonian
# =========================
H = -0.5*L + V_matrix

# =========================
# Solve eigenvalues/eigenvectors
# =========================
num_states = 4  # lowest 4 eigenstates
energies, wavefuncs = eigsh(H, k=num_states, sigma=0.0, which='LM')


# Print energies
print(f"Lowest {num_states} energies:", energies)

# =========================
# Visualization (slices)
# =========================
slice_index = N//2  # middle slice along z-axis

for i in range(num_states):
    # Reshape full 3D wavefunction to (N, N, N)
    psi_3D = wavefuncs[:, i].reshape(N, N, N)
    
    # Take the middle z slice
    psi_slice = psi_3D[:, :, slice_index]
    
    # Plot the slice
    plt.imshow(psi_slice, extent=[x_min, x_max, x_min, x_max],
               origin='lower', cmap='viridis')
    plt.title(f"State {i}, E={energies[i]:.3f}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='ψ(x,y,z=0)')
    plt.show()
