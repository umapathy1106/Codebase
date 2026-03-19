import numpy as np

# ── Load data ────────────────────────────────────────────────
data_dir = "HW2_CSV_DATA"
A = np.loadtxt(f"{data_dir}/HW2_A.csv")
y = np.loadtxt(f"{data_dir}/HW2_y.csv")
xinit = np.loadtxt(f"{data_dir}/HW2_xinit.csv")

N, K = A.shape  # N = number of rows, K = number of columns

print(f"A shape : {A.shape}  →  N = {N}, K = {K}")
print(f"y shape : {y.shape}")
print(f"xinit shape: {xinit.shape}")

# ── Least-squares loss function  L(x) = ||Ax - y||_2^2 ──────
def loss(x):
    r = A @ x - y # residual vector (N x 1)
    return r @ r # This computes the squared L2 norm of the residual, which is the least-squares loss.

# ══════════════════════════════════════════════════════════════
# Problem 2: Computations
# ══════════════════════════════════════════════════════════════

# ── Task 1: Compute H and C ─────────────────────────────────
H = 2 * A.T @ A                          # Hessian (K x K)
print(f"\n--- Task 1: H and C ---")
print(f"H shape    : {H.shape}")
print(f"H (first 5x5 block):\n{H[:5, :5]}\n")

# Lipschitz constant C = 2 * sigma_max^2
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
# Note: sigma_max is the largest singular value of A, so C = 2 * sigma_max^2
# Alternatively, we could compute C as the spectral norm of H, which is also 2 * sigma_max^2.
C = 2 * sigma_max**2 # Lipschitz constant for the gradient of L(x)
print(f"sigma_max  : {sigma_max:.6f}")
print(f"C          : {C:.6f}")

# ── Task 2: LS solution via SVD ──────────────────────────────
U, s, Vt = np.linalg.svd(A, full_matrices=True) # U (N x N), s (min(N, K) values), Vt (K x K)
V = Vt.T # V is the transpose of Vt, so V (K x K)
# The numerical rank r of A is the number of singular values that are greater than a small threshold (e.g., 1e-10).
r = np.sum(s > 1e-10)                    # numerical rank of A (number of non-zero singular values)

print(f"\n--- Task 2: SVD-based LS solution ---")
print(f"SVD: U {U.shape}, s ({s.shape[0]} values), V {V.shape}")
print(f"rank(A)    : {r}")

# xopt = V Sigma^+ U^T y = sum_{i=1}^{r} (u_i^T y / sigma_i) * v_i
UTy = U.T @ y # U^T y (N x 1)
xopt_svd = np.zeros(K) # initialize xopt as a zero vector of length K

# Compute the SVD-based least-squares solution using only the non-zero singular values
for i in range(r):
    xopt_svd += (UTy[i] / s[i]) * V[:, i] # This is the contribution of the i-th singular value to the solution, where UTy[i] is the i-th component of U^T y, s[i] is the i-th singular value, and V[:, i] is the i-th right singular vector.

# Lopt = sum_{i=r+1}^{N} (u_i^T y)^2
# The optimal loss Lopt is the sum of squares of the components of U^T y corresponding to the zero singular values, which are the last N-r components of U^T y.
Lopt = np.sum(UTy[r:]**2) # This is the optimal loss value, which is the sum of squares of the components of U^T y corresponding to the zero singular values.

print(f"xopt (first 5 entries): {xopt_svd[:5]}")
print(f"\nL(xinit)   : {loss(xinit):.6f}")
print(f"L(xopt)    : {loss(xopt_svd):.6f}")
print(f"Lopt (SVD) : {Lopt:.6f}")

# ── Verification: compare SVD solution with lstsq ────────────
xopt_lstsq = np.linalg.lstsq(A, y, rcond=None)[0] # np.linalg.lstsq returns a tuple (xopt, residuals, rank, singular_values), we only need xopt
print(f"\n--- Verification ---")
print(f"||xopt_svd - xopt_lstsq|| = {np.linalg.norm(xopt_svd - xopt_lstsq):.2e}")

# ══════════════════════════════════════════════════════════════
# Problem 3: LS GD with Exact Line Search (ELS)
# ══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

# Gradient of L(x): g(x) = 2 A^T (Ax - y)
# Note: The factor of 2 comes from the derivative of the squared norm. The gradient points in the direction of steepest ascent, so we will move in the opposite direction for gradient descent.
def gradient(x):
    """Gradient of L(x) = ||Ax - y||^2: g(x) = 2 A^T (Ax - y)"""
    return 2 * A.T @ (A @ x - y) # This computes the gradient of the least-squares loss function at the point x.

# ── GD with ELS: 30 iterations ──────────────────────────────
num_iters = 30 # number of iterations for gradient descent
x = xinit.copy() # initialize x with the given xinit, we will update this variable in-place during the iterations
loss_history = [] # to store the loss values at each iteration
gamma_history = [] # to store the step sizes (gamma_n) at each iteration

# The exact line search (ELS) step size gamma_n is computed as:
# gamma_n = (g^T g) / (g^T H g) = ||g||^2 / (2 * ||Ag||^2)
# This step size minimizes the loss along the direction of the negative gradient at each iteration, ensuring the most efficient descent towards the minimum.
for n in range(num_iters):
    g = gradient(x) # compute the gradient at the current point x
    # ELS step size: gamma = g^T g / (g^T H g) = ||g||^2 / (2 * ||Ag||^2)
    Ag = A @ g # compute A g, which is needed for the denominator of the ELS step size
    gamma_n = (g @ g) / (2 * (Ag @ Ag))
    x = x - gamma_n * g # update x using the computed step size and the gradient
    loss_history.append(loss(x)) # compute and store the loss at the new point x
    gamma_history.append(gamma_n) # store the step size used at this iteration

# ── Fig 1: L(xn) vs iteration ───────────────────────────────
iters = np.arange(1, num_iters + 1)

# The first figure plots the loss L(xn) at each iteration of the gradient descent with exact line search. We also include a horizontal dashed line to indicate the optimal loss Lopt obtained from the SVD solution. This allows us to visually assess how quickly the gradient descent method converges towards the optimal solution.
plt.figure(figsize=(8, 5))
plt.plot(iters, loss_history, 'b-o', markersize=4, label='GD with ELS')
plt.axhline(y=Lopt, color='k', linestyle='--', label=f'$L_{{opt}}$ = {Lopt:.2f}')
plt.xlabel('Iteration n')
plt.ylabel('L($x_n$)')
plt.title('Fig 1: Loss vs Iteration (GD with ELS)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_loss_vs_iteration.png', dpi=150)
plt.close()
print("Saved Fig 1 to fig1_loss_vs_iteration.png")

# ── Fig 2: gamma_n vs iteration ─────────────────────────────
# The second figure plots the step size gamma_n at each iteration of the gradient descent with exact line search. We also include a horizontal dashed line to indicate the value 1/C, which serves as a reference for the step size. This allows us to visually assess how the step size evolves over the iterations.
plt.figure(figsize=(8, 5))
plt.plot(iters, gamma_history, 'b-o', markersize=4, label='$\\gamma_n$ (ELS)')
plt.axhline(y=1/C, color='k', linestyle='--', label=f'1/C = {1/C:.6f}')
plt.xlabel('Iteration n')
plt.ylabel('$\\gamma_n$')
plt.title('Fig 2: Step Size vs Iteration (GD with ELS)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_stepsize_vs_iteration.png', dpi=150)
plt.close()
print("Saved Fig 2 to fig2_stepsize_vs_iteration.png")

# ── Print summary ────────────────────────────────────────────
print(f"\n--- Problem 3 Summary ---")
print(f"L(x0)  = {loss(xinit):.6f}")
print(f"L(x30) = {loss_history[-1]:.6f}")
print(f"Lopt   = {Lopt:.6f}")
print(f"1/C    = {1/C:.6f}")
print(f"gamma range: [{min(gamma_history):.6f}, {max(gamma_history):.6f}]")

# ══════════════════════════════════════════════════════════════
# Problem 4: LS GD with Fixed Step Size (FSS)
# ══════════════════════════════════════════════════════════════

# In this part, we will run gradient descent with fixed step sizes of the form gamma = p / C, where p is a positive scalar. We will test several values of p to see how it affects the convergence of the algorithm. The step size gamma is chosen to be proportional to 1/C, which is a common choice for fixed step sizes in gradient descent, as it ensures that the step size is appropriately scaled with respect to the Lipschitz constant of the gradient.
p_values = [0.1, 0.5, 1.0, 1.5, 2.0]
colors = ['green', 'blue', 'red', 'orange', 'purple']

plt.figure(figsize=(10, 6))

for p, color in zip(p_values, colors):
    gamma_fixed = p * (1 / C)             # fixed step size = p / C
    x = xinit.copy()
    loss_fss = []
    for n in range(num_iters):
        g = gradient(x)
        x = x - gamma_fixed * g
        loss_fss.append(loss(x))
    plt.plot(iters, loss_fss, '-o', markersize=3, color=color,
             label=f'p = {p} ($\\gamma$ = {gamma_fixed:.6f})')
    print(f"p = {p:.1f}: gamma = {gamma_fixed:.6f}, L(x30) = {loss_fss[-1]:.6f}")

plt.axhline(y=Lopt, color='k', linestyle='--', linewidth=1.5, label=f'$L_{{opt}}$ = {Lopt:.2f}')
plt.xlabel('Iteration n')
plt.ylabel('L($x_n$)')
plt.title('Fig 3: Loss vs Iteration (GD with Fixed Step Size)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('fig3_fss_loss_vs_iteration.png', dpi=150)
plt.close()
print("Saved Fig 3 to fig3_fss_loss_vs_iteration.png")

# ══════════════════════════════════════════════════════════════
# Problem 5: LS GD with Backtracking Line Search (BLS)
# ══════════════════════════════════════════════════════════════
import time

def gd_bls(A, y, xinit, num_iters, eta, c, C):
    """
    GD with Backtracking Line Search.
    BLS: start with gamma = 1 (large initial guess), then while the Armijo condition
         L(x - gamma*g) > L(x) - c*gamma*||g||^2
    is violated, shrink gamma *= eta.
    Returns loss_history, gamma_history, and elapsed time.
    """
    x = xinit.copy()
    loss_history = []
    gamma_history = []

    t_start = time.time()
    for n in range(num_iters):
        g = gradient(x)
        g_norm_sq = g @ g
        Lx = loss(x)
        gamma = 1.0  # start with large initial step size

        # Backtracking: shrink gamma until Armijo condition is satisfied
        while loss(x - gamma * g) > Lx - c * gamma * g_norm_sq:
            gamma *= eta

        x = x - gamma * g
        loss_history.append(loss(x))
        gamma_history.append(gamma)
    t_end = time.time()

    return loss_history, gamma_history, t_end - t_start

# ── Fig 4: L(xn) vs n, c=0.1, varying eta ───────────────────
eta_values = [0.1, 0.2, 0.5, 0.9]
c_fixed = 0.1
colors4 = ['green', 'blue', 'red', 'orange']

plt.figure(figsize=(10, 6))
results_fig4 = {}
for eta, color in zip(eta_values, colors4):
    lh, gh, rt = gd_bls(A, y, xinit, num_iters, eta, c_fixed, C)
    results_fig4[eta] = (lh, gh, rt)
    plt.plot(iters, lh, '-o', markersize=3, color=color,
             label=f'$\\eta$ = {eta}')
    print(f"Fig4: eta={eta}, c={c_fixed}: L(x30)={lh[-1]:.6f}, runtime={rt:.4f}s")

plt.axhline(y=Lopt, color='k', linestyle='--', linewidth=1.5, label=f'$L_{{opt}}$ = {Lopt:.2f}')
plt.xlabel('Iteration n')
plt.ylabel('L($x_n$)')
plt.title('Fig 4: Loss vs Iteration (GD with BLS, c = 0.1)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_bls_loss_vs_eta.png', dpi=150)
plt.close()
print("Saved Fig 4 to fig4_bls_loss_vs_eta.png")

# ── Fig 5: Runtime bars vs eta (c=0.1) ──────────────────────
runtimes_eta = [results_fig4[eta][2] for eta in eta_values]
eta_labels = [str(eta) for eta in eta_values]

plt.figure(figsize=(8, 5))
bars = plt.bar(eta_labels, runtimes_eta, color=colors4, edgecolor='black')
for bar, rt in zip(bars, runtimes_eta):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{rt:.4f}s', ha='center', va='bottom', fontsize=10)
plt.xlabel('$\\eta$')
plt.ylabel('Runtime (seconds)')
plt.title('Fig 5: Runtime vs $\\eta$ (GD with BLS, c = 0.1)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig5_bls_runtime_vs_eta.png', dpi=150)
plt.close()
print("Saved Fig 5 to fig5_bls_runtime_vs_eta.png")

# ── Fig 6: gamma_n vs n, c=0.1, varying eta ─────────────────
plt.figure(figsize=(10, 6))
for eta, color in zip(eta_values, colors4):
    gh = results_fig4[eta][1]
    plt.plot(iters, gh, '-o', markersize=3, color=color,
             label=f'$\\eta$ = {eta}')

plt.axhline(y=1/C, color='k', linestyle='--', linewidth=1.5, label=f'1/C = {1/C:.6f}')
plt.xlabel('Iteration n')
plt.ylabel('$\\gamma_n$')
plt.title('Fig 6: Step Size vs Iteration (GD with BLS, c = 0.1)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig6_bls_gamma_vs_eta.png', dpi=150)
plt.close()
print("Saved Fig 6 to fig6_bls_gamma_vs_eta.png")

# ── Fig 7: L(xn) vs n, eta=0.5, varying c ──────────────────
c_values = [0.01, 0.1]
eta_fixed = 0.5
colors7 = ['blue', 'red']

plt.figure(figsize=(10, 6))
results_fig7 = {}
for c_val, color in zip(c_values, colors7):
    lh, gh, rt = gd_bls(A, y, xinit, num_iters, eta_fixed, c_val, C)
    results_fig7[c_val] = (lh, gh, rt)
    plt.plot(iters, lh, '-o', markersize=3, color=color,
             label=f'c = {c_val}')
    print(f"Fig7: eta={eta_fixed}, c={c_val}: L(x30)={lh[-1]:.6f}, runtime={rt:.4f}s")

plt.axhline(y=Lopt, color='k', linestyle='--', linewidth=1.5, label=f'$L_{{opt}}$ = {Lopt:.2f}')
plt.xlabel('Iteration n')
plt.ylabel('L($x_n$)')
plt.title('Fig 7: Loss vs Iteration (GD with BLS, $\\eta$ = 0.5)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig7_bls_loss_vs_c.png', dpi=150)
plt.close()
print("Saved Fig 7 to fig7_bls_loss_vs_c.png")

# ── Fig 8: Runtime bars vs c (eta=0.5) ──────────────────────
runtimes_c = [results_fig7[c_val][2] for c_val in c_values]
c_labels = [str(c_val) for c_val in c_values]

plt.figure(figsize=(8, 5))
bars = plt.bar(c_labels, runtimes_c, color=colors7, edgecolor='black')
for bar, rt in zip(bars, runtimes_c):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{rt:.4f}s', ha='center', va='bottom', fontsize=10)
plt.xlabel('c')
plt.ylabel('Runtime (seconds)')
plt.title('Fig 8: Runtime vs c (GD with BLS, $\\eta$ = 0.5)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig8_bls_runtime_vs_c.png', dpi=150)
plt.close()
print("Saved Fig 8 to fig8_bls_runtime_vs_c.png")

# ── Fig 9: gamma_n vs n, eta=0.5, varying c ─────────────────
plt.figure(figsize=(10, 6))
for c_val, color in zip(c_values, colors7):
    gh = results_fig7[c_val][1]
    plt.plot(iters, gh, '-o', markersize=3, color=color,
             label=f'c = {c_val}')

plt.axhline(y=1/C, color='k', linestyle='--', linewidth=1.5, label=f'1/C = {1/C:.6f}')
plt.xlabel('Iteration n')
plt.ylabel('$\\gamma_n$')
plt.title('Fig 9: Step Size vs Iteration (GD with BLS, $\\eta$ = 0.5)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig9_bls_gamma_vs_c.png', dpi=150)
plt.close()
print("Saved Fig 9 to fig9_bls_gamma_vs_c.png")
