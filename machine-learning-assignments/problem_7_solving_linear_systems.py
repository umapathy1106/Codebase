import numpy as np

# ---------------------------------------------------------
# Problem 7: Solving Linear Systems
# ---------------------------------------------------------

# Define matrices
A = np.array([
    [-2.74125009,  2.24215689, -0.60553211, -0.16755625],
    [-0.34868395,  0.29538923, -0.45259498,  0.50015934],
    [ 2.49664208,  0.27798324,  2.00739274,  0.2197803 ]
])

B = np.array([
    [-2.74125009, -0.34868395,  2.49664208],
    [ 2.24215689,  0.29538923,  0.27798324],
    [-0.60553211, -0.45259498,  2.00739274],
    [-0.16755625,  0.50015934,  0.2197803 ]
])

C = np.array([
    [ 0.31997336,  0.43316234, -0.33457014, -0.34017903],
    [ 1.12969075,  1.52931319, -1.18122581, -1.20102843],
    [ 0.2008776,   0.27193705, -0.21004138, -0.21356262]
])

D = np.array([
    [ 0.07999334,  0.28242269,  0.0502194 ],
    [ 0.10829058,  0.3823283,   0.06798426],
    [-0.08364254, -0.29530645, -0.05251035],
    [-0.08504476, -0.30025711, -0.05339065]
])

# Define right-hand side vectors
yA  = np.array([ 0.61339829,  0.11012282, -0.06426754])

yB  = np.array([ 0.66761214,  0.35931116,  0.74289966,  0.02979187])
yB2 = np.array([ 0.24982762, -0.45768269,  0.22778277,  0.6341392 ])

yC  = np.array([ 0.1421664,   0.50192948,  0.08925132])
yC2 = np.array([-1.01480112,  0.4115211,  -0.45229071])

yD  = np.array([ 0.41615372,  0.56336601, -0.43513813, -0.44243299])
yD2 = np.array([ 0.47277025, -0.64357627,  1.30059591,  1.426948  ])

# ---------------------------------------------------------
# Define the 7 cases
# ---------------------------------------------------------
cases = [
    ("Case 1", A, yA,  "G = A, f = yA"),
    ("Case 2", B, yB,  "G = B, f = yB"),
    ("Case 3", B, yB2, "G = B, f = yB2"),
    ("Case 4", C, yC,  "G = C, f = yC"),
    ("Case 5", C, yC2, "G = C, f = yC2"),
    ("Case 6", D, yD,  "G = D, f = yD"),
    ("Case 7", D, yD2, "G = D, f = yD2"),
]


def analyze_linear_system(case_name, G, f, description):
    """Analyze the linear system Gw = f."""
    print("\n" + "=" * 70)
    print(f"{case_name}: {description}")
    print("=" * 70)

    m, n = G.shape
    print(f"\n1. SIZE AND RANK:")
    print(f"   G shape: {m} × {n} (m={m} rows, n={n} columns)")

    rank_G = np.linalg.matrix_rank(G)
    print(f"   rank(G) = {rank_G}")
    print(f"   nullity(G) = n - rank = {n} - {rank_G} = {n - rank_G}")

    # 2. Orthogonal projection matrix onto Range(G)
    # P = G @ G⁺  (m × m projection matrix)
    print(f"\n2. ORTHOGONAL PROJECTION ONTO Range(G):")
    G_pinv = np.linalg.pinv(G)
    P = G @ G_pinv  # m × m projection matrix
    print(f"   P = G · G⁺, shape: {P.shape}")
    print(f"   P =")
    for row in P:
        print(f"     [{', '.join(f'{v:10.6f}' for v in row)}]")

    # Verify P is a projection: P² = P and Pᵀ = P
    is_idempotent = np.allclose(P @ P, P, atol=1e-10)
    is_symmetric = np.allclose(P, P.T, atol=1e-10)
    print(f"   Verification: P² = P? {is_idempotent}, P = Pᵀ? {is_symmetric}")

    # 3. Check if f ∈ Range(G)
    # f ∈ Range(G) iff Pf = f
    print(f"\n3. IS f ∈ Range(G)?")
    Pf = P @ f
    residual_proj = np.linalg.norm(Pf - f)
    f_in_range = np.allclose(Pf, f, atol=1e-8)
    print(f"   Pf = [{', '.join(f'{v:.8f}' for v in Pf)}]")
    print(f"   f  = [{', '.join(f'{v:.8f}' for v in f)}]")
    print(f"   ||Pf - f|| = {residual_proj:.2e}")
    print(f"   f ∈ Range(G): {f_in_range}")

    # 4. Overdetermined or underdetermined?
    print(f"\n4. SYSTEM TYPE:")
    if m > n:
        system_type = "OVERDETERMINED"
        print(f"   m={m} > n={n} → {system_type} (more equations than unknowns)")
    elif m < n:
        system_type = "UNDERDETERMINED"
        print(f"   m={m} < n={n} → {system_type} (fewer equations than unknowns)")
    else:
        system_type = "SQUARE"
        print(f"   m={m} = n={n} → {system_type}")
        if rank_G < n:
            print(f"   But rank(G) = {rank_G} < {n}, so the system is RANK-DEFICIENT")

    # 5. Solution existence and uniqueness
    print(f"\n5. SOLUTION ANALYSIS:")
    G_aug = np.column_stack([G, f])
    rank_aug = np.linalg.matrix_rank(G_aug)
    print(f"   rank(G) = {rank_G}, rank([G|f]) = {rank_aug}")

    if rank_G < rank_aug:
        solution_type = "NO SOLUTION"
        print(f"   rank(G) < rank([G|f]) → {solution_type}")
        print(f"   The system Gw = f is INCONSISTENT.")
    elif rank_G == n:
        solution_type = "UNIQUE SOLUTION"
        print(f"   rank(G) = rank([G|f]) = n → {solution_type}")
    else:
        solution_type = "INFINITELY MANY SOLUTIONS"
        print(f"   rank(G) = rank([G|f]) < n → {solution_type}")
        print(f"   The solution space is a ({n - rank_G})-dimensional affine subspace.")

    # 6. Compute a solution
    print(f"\n6. SOLUTION:")
    if solution_type == "NO SOLUTION":
        # Least-squares solution: w = G⁺f (minimizes ||Gw - f||)
        w = G_pinv @ f
        residual = np.linalg.norm(G @ w - f)
        print(f"   No exact solution exists.")
        print(f"   Least-squares solution (w = G⁺f, minimizes ||Gw - f||):")
        print(f"   w = [{', '.join(f'{v:.8f}' for v in w)}]")
        print(f"   ||Gw - f|| = {residual:.8f}")
    elif solution_type == "UNIQUE SOLUTION":
        if m == n and rank_G == n:
            w = np.linalg.solve(G, f)
        else:
            w = G_pinv @ f
        residual = np.linalg.norm(G @ w - f)
        print(f"   Unique solution:")
        print(f"   w = [{', '.join(f'{v:.8f}' for v in w)}]")
        print(f"   Verification: ||Gw - f|| = {residual:.2e}")
    else:
        # Minimum-norm solution: w = G⁺f
        w = G_pinv @ f
        residual = np.linalg.norm(G @ w - f)
        print(f"   Infinitely many solutions exist.")
        print(f"   Minimum-norm solution (w = G⁺f, minimizes ||w||):")
        print(f"   w = [{', '.join(f'{v:.8f}' for v in w)}]")
        print(f"   ||w|| = {np.linalg.norm(w):.8f}")
        print(f"   Verification: ||Gw - f|| = {residual:.2e}")

        # Show the null space basis
        U_svd, S_svd, Vt_svd = np.linalg.svd(G)
        null_dims = n - rank_G
        null_basis = Vt_svd[-null_dims:, :].T
        print(f"\n   Null space basis (columns of N):")
        for j in range(null_dims):
            print(f"     n_{j+1} = [{', '.join(f'{v:.8f}' for v in null_basis[:, j])}]")
        print(f"   General solution: w = w_particular + Σ αᵢnᵢ  (αᵢ ∈ ℝ)")

    return w if solution_type != "NO SOLUTION" else G_pinv @ f


# ---------------------------------------------------------
# Run all 7 cases
# ---------------------------------------------------------
print("*" * 70)
print("PROBLEM 7: SOLVING LINEAR SYSTEMS — Gw = f")
print("*" * 70)

for case_name, G, f, desc in cases:
    analyze_linear_system(case_name, G, f, desc)

# ---------------------------------------------------------
# Summary Table
# ---------------------------------------------------------
print("\n\n" + "=" * 95)
print("SUMMARY TABLE")
print("=" * 95)
print(f"{'Case':<8} | {'Size':<8} | {'Rank':<5} | {'Type':<16} | {'f ∈ R(G)?':<10} | {'Solution':<25}")
print("-" * 95)

for case_name, G, f, desc in cases:
    m, n = G.shape
    rank_G = np.linalg.matrix_rank(G)
    P = G @ np.linalg.pinv(G)
    f_in_range = np.allclose(P @ f, f, atol=1e-8)
    G_aug = np.column_stack([G, f])
    rank_aug = np.linalg.matrix_rank(G_aug)

    if m > n:
        sys_type = "Overdetermined"
    elif m < n:
        sys_type = "Underdetermined"
    else:
        sys_type = "Square"

    if rank_G < rank_aug:
        sol = "No solution (LS)"
    elif rank_G == n:
        sol = "Unique"
    else:
        sol = "Infinitely many"

    print(f"{case_name:<8} | {m}×{n:<5} | {rank_G:<5} | {sys_type:<16} | {str(f_in_range):<10} | {sol:<25}")
