# Formulation

The controller as implemented: layer 1 (contact forces over a horizon, `controller/dist_horizon.py`), the
obstacle keep-out added to it, the steering reference shaper (`run()` in `controller/koopman_admm.py`), and
layer 2 (Koopman swing legs). Parameter values are the defaults in the code.

## Notation

```
N = 8 horizon steps, Δ = 0.02 s; control tick 5 ms (the warm start is shifted by 0.25 step)
feet i ∈ {1..4};  a_ik ∈ {0,1}: foot i in stance at step k
f_ik ∈ R³                      contact force, world frame
G_ik = [ I₃ ; [r_ik]× ]        grasp map,  r_ik = foot_ik − com_k
W_k  = Σ_i a_ik G_ik f_ik      wrench on the trunk at step k
x_k  = [v_k ; ω_k] ∈ R⁶         trunk linear and angular velocity, world frame
P_k  = [p_k ; Θ_k] ∈ R⁶         trunk position and small-angle attitude (absolute yaw)
X_k  = [P_k ; x_k] ∈ R¹²
m = 12.74 kg,  I_W = R_z(ψ) diag(0.1268, 0.394, 0.4229) R_z(ψ)ᵀ,  g = 9.81 m/s²
```

## Layer 1: contact forces over the horizon

Single rigid body, wrench constant over each step:

```
x_{k+1} = x_k + B_d W_k + c_d,     B_d = blkdiag(Δ/m · I₃, Δ · I_W⁻¹),     c_d = [−Δ g e_z ; 0]
x_{1..N} = S_w W + base,           S_w = tril(1_N) ⊗ B_d,     base_k = x_0 + (k+1) c_d
```

References, from the commands and the measured state:

```
v_ref = R_z(ψ) [v_x, v_y, 0]ᵀ + 25 (z₀ − z) e_z,                    z₀ = 0.27 m
ω_ref = [0, 0, ω_z + ω_ff]ᵀ − K_θ e_θ,                              K_θ = diag(40, 40, 3.5)
e_θ   = R_z(ψ) [φ, θ, 0]ᵀ + [0, 0, e_ψ]ᵀ,   ē_ψ = wrap(ψ − ψ_ref − ψ_off),   e_ψ = ē_ψ + 1.5 ∫ē_ψ dt  (integral clipped to ±0.4)
```

Cost and per-foot constraints:

```
J(f) = Σ_{k=0}^{N−1} ½ ‖x_{k+1} − x_ref‖²_Q + (λ/2) Σ_{i,k} a_ik ‖f_ik − f_nom‖²
Q = diag(400, 400, 400, 30, 30, 15),   λ = 0.005,   f_nom = (0, 0, mg/4)

|f_x| ≤ μ f_z,  |f_y| ≤ μ f_z,  0 ≤ f_z ≤ mg     (μ = 0.7)
f_ik = 0 when a_ik = 0
```

## Pose prediction (for the keep-out)

With the wrench piecewise constant the trapezoid rule is exact for position:

```
P_{k+1} = P_k + (Δ/2)(x_k + x_{k+1})      ⇒      X_{k+1} = S_x[k] W + x_b[k]
∂p_{k+1}/∂W_j = Δ (k − j + ½) B_d,v   for j ≤ k
```

Attitude uses Θ̇ ≈ ω (small angles, no gyroscopic term, I_W frozen at the current heading).

## Keep-out constraint

For an obstacle o (centre c_o, radius r_o) and a body sphere b (offset b_b in the trunk frame, radius r_b):

```
g_kob(X_{k+1}) = r_o + r_b + h_kob − ‖ M_o ( p_{k+1} + R(Θ_{k+1}) b_b − c_o ) ‖ ≤ 0
M_o = diag(1, 1, 0) for a vertical cylinder, I₃ for a sphere
body spheres: b_b ∈ {(0.19, 0, 0), (0, 0, 0), (−0.19, 0, 0)} m, r_b = 0.17 m
```

Linearized each tick at X̄, the previous plan moved on by 0.25 step (a constant-twist guess on the first tick):

```
r̄_w = R_z(ψ̄) b_b,     d̄ = M_o (p̄ + r̄_w − c_o),     n = d̄ / ‖d̄‖
ℓ   = (r̄_w × n) ⊙ (1, 1, 0)                     roll and pitch lever arm; the heading is held at ψ̄

−nᵀ p_{k+1} − ℓᵀ Θ_{k+1}  ≤  −nᵀ c_o − r_o − r_b + nᵀ r̄_w − h_kob
```

This is the tangent half-space: since ‖n‖ = 1, nᵀ(q − c) ≥ r implies ‖M(q − c)‖ ≥ r for the body-sphere
centre q, whatever the linearization point. Rotating the spheres with roll and pitch is linearized about
level; the fixed margin covers that error. Holding the heading keeps the plan from turning the trunk to
creep closer to an obstacle.

## Tightening (GPU-SLS tube)

A disturbance w_j with ‖w_j‖₂ ≤ 1 per step enters as a wrench E_W w_j, plus an initial-state error E₀ w₀:

```
E_W = diag(σ_f I₃, σ_τ I₃),  σ_f = 20 N,  σ_τ = 3 N·m;      E₀ = diag(σ_p I₆, σ_v I₆)   (0 in simulation)

h_kob = Σ_{j≤k} ‖ cᵀ (∂X_{k+1}/∂W_j) E_W ‖₂ + ‖ cᵀ (∂X_{k+1}/∂X₀) E₀ ‖₂ + δ,      δ = 0.02 m
```

with c the row's coefficient vector. This is gpu_sls `get_betas` and `get_constraint_tightenings` with
Φ_u = 0 (the open-loop response: only the first step is applied before the next solve). For a position row:

```
h_k = σ_f Δ² (k+1)² / (2m) + δ          2.0 cm at the first step, 4.0 cm at the last
```

A row is used only when its slack at X̄ is below 0.30 m and k ≥ 1 (the first step can move the trunk by
2 to 6 mm). Rows are soft: an exact penalty L·max(0, row − limit), L = 10⁵, which gives the hard solution
whenever that is feasible with multipliers below L (they reached about 4·10⁴ in the scenarios).

## ADMM

```
min  Σ_ik ψ_ik(f_ik) + Φ(Σ_i s_i) + Σ_r L·max(0, y_r − h_r)
s.t. G_ik f_ik = s_ik,   y = C_W Σ_i s_i

ψ_ik(f) = (λ/2)‖f − f_nom‖² + ι_pyramid(f)
Φ(W)    = ½ Wᵀ Q_W W + q_Wᵀ W,     Q_W = S_wᵀ Q̄ S_w,     q_W = S_wᵀ Q̄ (base − ref)
C_W     = C_row S_x  (rows on the stacked wrench),     h = row limits − C_row x_b
```

One iteration; scaled duals l (feet) and ν (rows); relaxation α = 1.6; ρ = 0.1 tr(Q_W)/(6N);
R_c = diag(ρ / ‖C_W,r‖²):

```
1. f_ik ← argmin ψ_ik(f) + (ρ/2)‖G_ik f − s_ik + l_ik‖²            exact, 19 active sets enumerated
2. z = C_W Σ s;   y ← min(z + ν, h) + max(z + ν − h − L/R_c, 0)
3. g_ik = α G_ik f_ik + (1 − α) s_ik;   y_r = α y + (1 − α) z
4. a_i = g_i + l_i,  A = Σ_i a_i
   ((Q_W + C_Wᵀ R_c C_W) N_k + ρ I) d = (Q_W + C_Wᵀ R_c C_W) A + q_W + C_Wᵀ R_c (ν − y_r)
   s_i ← a_i − d                                                     (N_k: stance count per step)
5. l_i ← l_i + g_i − s_i;   ν ← ν + C_W Σ s − y_r;   multipliers λ_c = R_c ν
```

(f, y) and s are the two ADMM blocks, so the fixed point is the constrained QP. 8 iterations per tick,
warm-started and aligned by contact pattern. Every iterate's forces are inside the pyramid; the rows hold at
convergence. Without rows the iteration is the original one.

## Steering

```
v_c = R_z(ψ_cmd) [v_x, v_y]ᵀ;    travel directions a_j = ∠v_c + δ_j,   δ_j ∈ linspace(−1.2, 1.2, 49) rad
body heading for a_j:            β_j = a_j − atan2(v_y, v_x)
swept trunk                      S_j = { p + s e(β_j) + t L e(a_j) : s ∈ [−0.19, 0.19], t ∈ [0, 1] },  L = 0.8 m
gap_j = min_o [ dist(c_o, S_j) − √((r_o + r_b + 0.06)² − Δz_o²) ]      (Δz = 0 for cylinders; spheres that
                                                                         do not reach the trunk height are ignored)
δ* = 0                                                if gap > 0 at δ = 0
     argmin_{gap_j > 0} |δ_j| + 0.5 |δ_j − ψ_off|      otherwise (hysteresis toward the current offset)
     the same over the least-blocked headings          if no gap_j > 0

ψ_off ← ψ_off + clip(δ* − ψ_off, ±0.8 Δt),     ω_ff = dψ_off/dt         (ψ_cmd = ψ_ref, Δt = 5 ms)
```

## Layer 2: Koopman swing legs

```
lift:   z = φ(q, q̇) = [q, q̇, sin q, cos q, sin(q₂ + q₃), cos(q₂ + q₃), q̇ ⊙ q̇] ∈ R¹⁷
model:  z⁺ = A z + B τ,     [A B] = Z⁺ [Z; U]ᵀ ([Z; U][Z; U]ᵀ + 10⁻⁵ I)⁻¹        (9,000 swing samples)
MPC:    min_τ Σ_{k=1}^{6} w_k ‖C_q z_k − q*‖² + 0.01 ‖C_q̇ z_k‖² + 5·10⁻⁵ ‖τ‖²,     w = (1, 1, 1, 1, 1, 8)
        s.t. |τ| ≤ (23.7, 23.7, 35.55) N·m on top of the bias torque
stance: τ_i = bias_i − J_iᵀ Rᵀ f_i
swing:  q* = IK of a Raibert foothold from the measured speed with a 6 cm clearance arc,
        rotated by (ω_z + ω_ff)·2T·(s − ½)
```

## Relation to GPU-SLS

| GPU-SLS (Fang and Chou, arXiv 2604.07644) | Here |
|---|---|
| g(x, u) ≤ 0 linearized every SQP iteration | the keep-out, linearized every 5 ms tick |
| h = Σ_j ‖rowᵀ (C Φ_x + D Φ_u)[k, j]‖₂ | the same, with Φ_u = 0 and Φ_x = (∂X/∂W) E_W |
| Φ from per-column LQR with FastSLS dual weights | open loop, since the feedback term is never applied |
| stagewise ADMM QP, parallel-scan Riccati on the GPU | the sharing ADMM with row copies; at N = 8 a scan gives nothing |
