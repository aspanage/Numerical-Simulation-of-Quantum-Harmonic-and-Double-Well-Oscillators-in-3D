import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.animation import FuncAnimation, PillowWriter

# =========================
# Grid and potential
# =========================
N = 200
x = np.linspace(-5,5,N)
dx = x[1]-x[0]

# 1D double-well potential
V = x**4 - 5*x**2

# =========================
# Finite difference Hamiltonian
# =========================
D = (-2*np.eye(N) + np.eye(N,k=1) + np.eye(N,k=-1)) / dx**2
H = -0.5*D + np.diag(V)

# =========================
# Solve eigenstates
# =========================
energies, wavefuncs = eigh(H)
psi0 = wavefuncs[:,0]
psi1 = wavefuncs[:,1]
E0 = energies[0]
E1 = energies[1]

# Normalize
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2)*dx)
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2)*dx)

# =========================
# Time evolution parameters
# =========================
t_max = 100
frames = 200
times = np.linspace(0, t_max, frames)

# =========================
# Set up figure
# =========================
fig, ax = plt.subplots(figsize=(8,5))
line, = ax.plot(x, np.abs((psi0+psi1)/np.sqrt(2))**2)
ax.set_ylim(0, np.max(np.abs(psi0+psi1)**2)*1.2)
ax.set_xlabel('x')
ax.set_ylabel('|Ψ(x,t)|^2')
ax.set_title('Quantum Tunneling Animation')

# =========================
# Animation function
# =========================
def animate(i):
    t = times[i]
    psi_t = (psi0*np.exp(-1j*E0*t) + psi1*np.exp(-1j*E1*t))/np.sqrt(2)
    line.set_ydata(np.abs(psi_t)**2)
    return line,

anim = FuncAnimation(fig, animate, frames=frames, interval=50)

# =========================
# Save animation
# =========================
save_as_gif = True
if save_as_gif:
    anim.save('double_well_tunneling.gif', writer=PillowWriter(fps=20))
    print("Saved animation as double_well_tunneling.gif")

plt.show()
