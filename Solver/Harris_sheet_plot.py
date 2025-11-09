import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
B0 = 1.0       # main magnetic field
a = 0.1        # thickness
nx = 200
ny = 200
Lx = 4
Ly = 2

# -----------------------------
# 1D PLOTS (y=0 slice)
# -----------------------------
x = np.linspace(-Lx/2, Lx/2, nx)

By = B0 * np.tanh(x / a)
Jz = (B0/a) * (1 / np.cosh(x/a)**2)  # analytic derivative

plt.figure()
plt.plot(x, By)
plt.xlabel("x")
plt.ylabel("B_y(x)")
plt.title("Harris Sheet Magnetic Field Profile")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, Jz)
plt.xlabel("x")
plt.ylabel("J_z(x)")
plt.title("Analytic Current Density of Harris Sheet")
plt.grid()
plt.show()

# -----------------------------
# 2D VECTOR FIELD WITH COLOR
# -----------------------------
X = np.linspace(-Lx/2, Lx/2, nx)
Y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(X, Y)

Bx = np.zeros_like(X)             
By2D = B0 * np.tanh(X / a)        

# magnitude
Bmag = np.sqrt(Bx**2 + By2D**2)

plt.figure(figsize=(6,4))
skip = 6
Q = plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               Bx[::skip, ::skip], By2D[::skip, ::skip],
               Bmag[::skip, ::skip],
               cmap='plasma', scale=40)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Harris Sheet Vector Field")
cb = plt.colorbar(Q)
cb.set_label(r"$|B|$")
plt.tight_layout()
plt.show()


