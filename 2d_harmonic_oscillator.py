import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh

# =========================
# Parameters
# =========================
N = 50                # Grid points per dimension
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Choose potential: 'harmonic' or 'double_well'
potential_type = 'double_well'

# =========================
# 1D Laplacian (finite difference)
# =========================
diag = -2.0 * np.ones(N)
off_diag = np.ones(N-1)
D = diags([off_diag, diag, off_diag], offsets=[-1,0,1]) / dx**2
I = eye(N)
L = kron(I, D) + kron(D, I)  # 2D Laplacian using Kronecker product

# =========================
# 2D Potential
# =========================
X, Y = np.meshgrid(x, x)

if potential_type == 'harmonic':
    V = 0.5 * (X**2 + Y**2)
elif potential_type == 'double_well':
    V = (X**4 + Y**4) - 5*(X**2 + Y**2)
else:
    raise ValueError("Unknown potential type.")

V_flat = V.flatten()
V_matrix = diags(V_flat, 0)

# =========================
# Hamiltonian
# =========================
H = -0.5 * L + V_matrix

# =========================
# Solve eigenvalues/eigenvectors
# =========================
num_states = 4  # number of lowest energy states
energies, wavefuncs = eigsh(H, k=num_states, which='SM')  # Smallest magnitude eigenvalues

# =========================
# Reshape for plotting
# =========================
wavefuncs_2D = [wavefuncs[:, i].reshape(N, N) for i in range(num_states)]

# =========================
# Plot eigenstates
# =========================
fig, axes = plt.subplots(2,2, figsize=(10,8))
for i, ax in enumerate(axes.flat):
    im = ax.imshow(wavefuncs_2D[i], extent=[x_min, x_max, x_min, x_max],
                   origin='lower', cmap='viridis')
    ax.set_title(f'n={i}, E={energies[i]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
plt.suptitle(f'2D {potential_type.replace("_", " ").title()} Potential Eigenstates', fontsize=16)
plt.tight_layout()
plt.show()

# =========================
# Print energies
# =========================
print(f"Lowest {num_states} energies:", energies)
