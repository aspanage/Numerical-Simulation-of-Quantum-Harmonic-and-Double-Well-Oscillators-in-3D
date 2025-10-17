import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh

# =========================
# Choose dimensionality: '1D' or '3D'
# =========================
dimension = '1D'

# =========================
# 1D Double-Well
# =========================
if dimension == '1D':
    N = 200
    x = np.linspace(-5,5,N)
    dx = x[1]-x[0]

    # Double-well potential
    V = x**4 - 5*x**2

    # Finite difference Laplacian
    D = (-2*np.eye(N) + np.eye(N,k=1) + np.eye(N,k=-1)) / dx**2

    # Hamiltonian
    H = -0.5 * D + np.diag(V)

    # Solve eigenvalues/eigenvectors
    energies, wavefuncs = eigh(H)

    # Normalize
    wavefuncs = wavefuncs / np.sqrt(dx)

    # Plot first 2 eigenstates
    plt.figure(figsize=(8,5))
    for n in range(2):
        plt.plot(x, wavefuncs[:,n] + energies[n], label=f'n={n}, E={energies[n]:.5f}')
    plt.xlabel('x')
    plt.ylabel('Energy + ψ(x)')
    plt.title('1D Double-Well Potential: Tunneling States')
    plt.legend()
    plt.show()

    # Energy splitting
    delta_E = energies[1] - energies[0]
    print(f"Energy splitting ΔE = {delta_E:.5f}")

# =========================
# 3D Double-Well
# =========================
elif dimension == '3D':
    N = 20  # keep small for memory
    x = np.linspace(-5,5,N)
    dx = x[1]-x[0]

    # 1D Laplacian
    diag = -2.0 * np.ones(N)
    off_diag = np.ones(N-1)
    D = diags([off_diag, diag, off_diag], offsets=[-1,0,1])
    I = eye(N)

    # 3D Laplacian
    L = kron(kron(I,I),D) + kron(kron(I,D),I) + kron(kron(D,I),I)

    # 3D double-well potential
    X, Y, Z = np.meshgrid(x,x,x,indexing='ij')
    V = X**4 + Y**4 + Z**4 - 5*(X**2 + Y**2 + Z**2)
    V_flat = V.flatten()
    V_matrix = diags(V_flat,0)

    # Hamiltonian
    H = -0.5*L + V_matrix

    # Solve lowest 2 states
    energies, wavefuncs = eigsh(H, k=2, which='SM')
    print(f"Lowest 2 energies: {energies}")
    print(f"Energy splitting ΔE = {energies[1]-energies[0]:.5f}")

    # Visualization: middle z-slice
    slice_index = N//2
    for i in range(2):
        psi_3D = wavefuncs[:,i].reshape(N,N,N)
        psi_slice = psi_3D[:,:,slice_index]
        plt.imshow(psi_slice, extent=[-5,5,-5,5], origin='lower', cmap='viridis')
        plt.title(f"State {i}, E={energies[i]:.5f}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='ψ(x,y,z=0)')
        plt.show()
