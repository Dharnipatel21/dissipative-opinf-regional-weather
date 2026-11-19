# Structure-Preserving Reduced-Order Modeling of Regional Atmospheric Dynamics over Tamil Nadu Using ERA5-Land

This work develops a non-intrusive reduced-order model (ROM) of the hourly near-surface atmosphere over Tamil Nadu, India, using ERA5-Land reanalysis data. The full state, consisting of five variables on 1065 land cells (5325 degrees of freedom), is compressed to ten modes by proper orthogonal decomposition (POD), which retains 92.1 percent of the snapshot energy. A quadratic manifold is compared with the linear POD subspace for reconstruction. The reduced dynamics are then learned from data by operator inference, in which a quadratic ordinary differential equation with known calendar forcing is fitted by regularised least squares and integrated with a fourth-order Runge-Kutta scheme. Two structural constraints are studied individually and jointly: an energy-preserving nonlinear term, enforced as a linear constraint on the quadratic operator, and a dissipative linear operator, enforced by decomposing it into a skew-symmetric part and a negative definite symmetric part. On the test period, the models with a dissipative linear operator remain bounded over a 30-day free run and give the lowest two-metre temperature error among the learned models at lead times from 3 to 36 hours, while the model with an energy-preserving nonlinear term alone does not remain bounded.

**Index Terms:** reduced order modeling, operator inference, proper orthogonal decomposition, quadratic manifold, energy-preserving dynamics, dissipative systems, ERA5-Land, regional weather forecasting, scientific machine learning.

---

## Introduction

Numerical weather prediction solves the governing equations of the atmosphere on fine grids at high computational cost. Data-driven reduced-order models replace the full state by a small number of coordinates and learn their evolution from data. Operator inference [1], [3] is a non-intrusive approach: it fits the reduced operators of a polynomial model directly to projected snapshot data, without access to the full-order solver. Because the advection terms of the governing equations are quadratic in the state, a quadratic reduced model is a natural structural choice.

A recurring difficulty is that unconstrained learned models can be unstable when integrated over long horizons. Structure-preserving formulations address this by building physical properties into the model, for example energy preservation of the nonlinear term [5] and Hamiltonian or Lagrangian structure [6], [7]. The near-surface atmosphere is forced and dissipative rather than Hamiltonian, so this work does not assume conservation of total energy. Instead it studies two constraints that are appropriate for forced dissipative systems: a nonlinear term that cannot create energy, and a linear operator that cannot amplify it.

The contributions of this repository are:

1. A reproducible pipeline that builds a state vector from ERA5-Land hourly fields over Tamil Nadu and compresses it with POD.
2. A comparison of the linear POD subspace and a quadratic manifold for state reconstruction.
3. An operator inference study with five model variants that isolate the effect of an energy-preserving nonlinear term and a dissipative linear operator.
4. A forecast and stability evaluation against persistence and hour-of-day climatology baselines on a held-out test period.

---

## II. Data

### A. Source

ERA5-Land hourly reanalysis [9], [10] on a 0.1 degree grid. The domain is bounded by 8.0 to 13.5 degrees N and 76.0 to 80.5 degrees E (56 by 46 grid points). Cells are retained if they are land cells inside the Tamil Nadu administrative boundary, which gives $N_c = 1065$ cells.

### B. Variables

| Symbol | Variable | Unit |
|---|---|---|
| t2m | 2 m temperature | degrees C |
| d2m | 2 m dew point temperature | degrees C |
| u10 | 10 m zonal wind | m/s |
| v10 | 10 m meridional wind | m/s |
| sp | Surface pressure | hPa |

Precipitation is excluded because it is intermittent and non-smooth, which is poorly suited to POD and polynomial reduced models.

### C. State vector and standardisation

Each variable $v$ is standardised using the mean $\mu_v$ and standard deviation $\sigma_v$ computed over the training period only. The state at hour $t$ stacks the five standardised fields over the $N_c$ cells:

$$
\mathbf{q}(t) \in \mathbb{R}^{n}, \qquad n = 5 N_c = 5325. \tag{1}
$$

### D. Temporal split

Whole calendar years are used so that no sample shares a period across splits.

| Split | Period | Hours used |
|---|---|---|
| Training | 2020 and 2022 | 16008 |
| Validation | 2023 | 8760 |
| Test | 2024 | 7368 |

Only hours present in the downloaded files are used. Training windows and forecast windows never span a missing hour. The number of forecast start times is 730 for validation (every 12 hours) and 1144 for testing (every 6 hours).

---

## Methodology

### A. Proper orthogonal decomposition

Let $\bar{\mathbf{q}}$ be the mean training state and $\mathbf{X} = [\mathbf{q}(t_1) - \bar{\mathbf{q}}, \dots, \mathbf{q}(t_m) - \bar{\mathbf{q}}]$ the centred snapshot matrix. The singular value decomposition is

$$
\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{W}^{\top}, \tag{2}
$$

and the POD basis $\mathbf{V}_r \in \mathbb{R}^{n \times r}$ contains the first $r$ columns of $\mathbf{U}$. The reduced coordinates and the linear reconstruction are

$$
\mathbf{z}(t) = \mathbf{V}_r^{\top} \big(\mathbf{q}(t) - \bar{\mathbf{q}}\big), \qquad \hat{\mathbf{q}}_{\mathrm{lin}} = \bar{\mathbf{q}} + \mathbf{V}_r \mathbf{z}. \tag{3}
$$

The fraction of snapshot energy captured by $r$ modes is

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^{2}}{\sum_{i=1}^{\min(n,m)} \sigma_i^{2}}. \tag{4}
$$

A randomised SVD is applied to every third training hour. The reduced dimension used for dynamics is $r = 10$.

### B. Quadratic manifold

A quadratic manifold [4] adds a curved correction to the linear subspace:

$$
\hat{\mathbf{q}}_{\mathrm{quad}} = \bar{\mathbf{q}} + \mathbf{V}_r \mathbf{z} + \mathbf{W}\, \mathrm{vech}\!\big(\mathbf{z}\mathbf{z}^{\top}\big), \tag{5}
$$

where $\mathrm{vech}(\cdot)$ stacks the $r(r+1)/2$ unique products $z_i z_j$ with $i \le j$. The matrix $\mathbf{W}$ is the ridge-regularised least-squares solution that maps the quadratic features to the residual of the linear reconstruction:

$$
\mathbf{W} = \arg\min_{\mathbf{W}} \; \big\| \mathbf{R} - \mathbf{W}\mathbf{F} \big\|_F^2 + \gamma \|\mathbf{W}\|_F^2, \tag{6}
$$

with $\mathbf{R} = (\mathbf{q} - \bar{\mathbf{q}}) - \mathbf{V}_r \mathbf{z}$ and $\mathbf{F}$ the matrix of scaled quadratic features.

### C. Reduced model by operator inference

The reduced state is assumed to follow a quadratic ordinary differential equation with known forcing:

$$
\dot{\mathbf{z}} = \mathbf{c} + \mathbf{A}\mathbf{z} + \mathbf{H}\,\mathrm{vech}\!\big(\mathbf{z}\mathbf{z}^{\top}\big) + \mathbf{B}\mathbf{u}(t), \tag{7}
$$

with $\mathbf{c} \in \mathbb{R}^{r}$, $\mathbf{A} \in \mathbb{R}^{r \times r}$, $\mathbf{H} \in \mathbb{R}^{r \times r(r+1)/2}$ and $\mathbf{B} \in \mathbb{R}^{r \times 4}$. The forcing vector encodes the hour of day $h$ and the day of year $d$:

$$
\mathbf{u}(t) = \Big[\sin\tfrac{2\pi h}{24},\; \cos\tfrac{2\pi h}{24},\; \sin\tfrac{2\pi d}{366},\; \cos\tfrac{2\pi d}{366}\Big]^{\top}. \tag{8}
$$

The time derivative is approximated by a central difference over three consecutive hours, with a step of one hour:

$$
\dot{\mathbf{z}}(t_k) \approx \tfrac{1}{2}\big(\mathbf{z}(t_{k+1}) - \mathbf{z}(t_{k-1})\big). \tag{9}
$$

Collecting the unknown operators in $\boldsymbol{\Theta}$ and the corresponding data features in $\mathbf{D}$, the operators solve the regularised least-squares problem

$$
\min_{\boldsymbol{\Theta}} \; \big\| \mathbf{D}\boldsymbol{\Theta} - \dot{\mathbf{Z}} \big\|_F^2 + \lambda \|\boldsymbol{\Theta}\|_F^2, \tag{10}
$$

with column equilibration of $\mathbf{D}$ and no penalty on the constant term. Forecasts are obtained by integrating Eq. (7) with the classical fourth-order Runge-Kutta method over a step $\Delta t = 1$ hour:

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + \tfrac{1}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big), \tag{11}
$$

where $\mathbf{k}_1 = f(\mathbf{z}_k, \mathbf{u}_k)$, $\mathbf{k}_2 = f(\mathbf{z}_k + \tfrac{1}{2}\mathbf{k}_1, \mathbf{u}_{k+1/2})$, $\mathbf{k}_3 = f(\mathbf{z}_k + \tfrac{1}{2}\mathbf{k}_2, \mathbf{u}_{k+1/2})$ and $\mathbf{k}_4 = f(\mathbf{z}_k + \mathbf{k}_3, \mathbf{u}_{k+1})$. The forcing is evaluated analytically at the start, middle and end of each step.

### D. Energy-preserving nonlinear term

Define the reduced energy $\mathcal{E}(\mathbf{z}) = \tfrac{1}{2}\|\mathbf{z}\|^2$. Because $\mathbf{V}_r$ has orthonormal columns, this is the energy of the reduced state. The contribution of the nonlinear term to $\dot{\mathcal{E}}$ is $\mathbf{z}^{\top}\mathbf{H}\,\mathrm{vech}(\mathbf{z}\mathbf{z}^{\top})$. The nonlinear term is energy preserving if

$$
\sum_{i=1}^{r} \sum_{j \le k} H_{i,jk}\, z_i z_j z_k = 0 \quad \text{for all } \mathbf{z} \in \mathbb{R}^{r}. \tag{12}
$$

Eq. (12) is a cubic polynomial in $\mathbf{z}$ that must vanish identically. This holds if and only if the total coefficient of every cubic monomial $z_a z_b z_c$ ($a \le b \le c$) is zero, which is a set of linear equality constraints on the entries of $\mathbf{H}$. Writing these constraints as $\mathbf{C}\,\mathrm{vec}(\mathbf{H}) = \mathbf{0}$ and taking an orthonormal basis $\mathbf{N}$ of the null space of $\mathbf{C}$,

$$
\mathrm{vec}(\mathbf{H}) = \mathbf{N}\mathbf{a}, \tag{13}
$$

the energy-preserving model is again a linear least-squares problem, now in the coefficients $\mathbf{a}$. For $r = 10$ the unconstrained operator has 550 entries, the constraints number 220, and the free parameter dimension is 330. A numerical check confirms that $\max |\mathbf{z}^{\top}\mathbf{H}\,\mathrm{vech}(\mathbf{z}\mathbf{z}^{\top})|$ is of the order of $10^{-14}$ for operators built from $\mathbf{N}$.

### E. Dissipative linear operator

The linear operator is decomposed as

$$
\mathbf{A} = \mathbf{J} - \mathbf{D}, \qquad \mathbf{J} = -\mathbf{J}^{\top}, \qquad \mathbf{D} = \mathbf{D}^{\top} \succ 0. \tag{14}
$$

Let $\mathbf{K} = \tfrac{1}{2}(\mathbf{A} - \mathbf{A}^{\top})$ and $\mathbf{S} = \tfrac{1}{2}(\mathbf{A} + \mathbf{A}^{\top}) = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}$. The nearest dissipative operator is obtained by clipping the eigenvalues of $\mathbf{S}$:

$$
\mathbf{A}_{\mathrm{dis}} = \mathbf{K} + \mathbf{Q}\,\min(\boldsymbol{\Lambda}, -\varepsilon)\,\mathbf{Q}^{\top}, \tag{15}
$$

with $\varepsilon = 10^{-3}$ per hour. The model is first fitted with a free $\mathbf{A}$, then $\mathbf{A}$ is replaced by $\mathbf{A}_{\mathrm{dis}}$, and the remaining operators $\mathbf{c}$, $\mathbf{H}$ and $\mathbf{B}$ are refitted with $\mathbf{A}_{\mathrm{dis}}$ held fixed.

With an energy-preserving $\mathbf{H}$ and a dissipative $\mathbf{A}$, the energy balance of Eq. (7) is

$$
\frac{d}{dt}\,\tfrac{1}{2}\|\mathbf{z}\|^2 = -\mathbf{z}^{\top}\mathbf{D}\mathbf{z} + \mathbf{z}^{\top}\big(\mathbf{c} + \mathbf{B}\mathbf{u}(t)\big) \le -\lambda_{\min}(\mathbf{D})\|\mathbf{z}\|^2 + \|\mathbf{c} + \mathbf{B}\mathbf{u}(t)\|\,\|\mathbf{z}\|. \tag{16}
$$

Since the forcing is bounded, Eq. (16) implies that $\limsup_{t \to \infty}\|\mathbf{z}(t)\| \le \sup_t \|\mathbf{c} + \mathbf{B}\mathbf{u}(t)\| / \lambda_{\min}(\mathbf{D})$.

### F. Model variants

| Model | Nonlinear term $\mathbf{H}$ | Linear operator $\mathbf{A}$ |
|---|---|---|
| `linear` | none | free |
| `quad` | free | free |
| `energy` | energy preserving, Eq. (12) | free |
| `linear_dis` | none | dissipative, Eq. (15) |
| `energy_dis` | energy preserving, Eq. (12) | dissipative, Eq. (15) |

### G. Regularisation selection

The regularisation weight $\lambda$ is chosen from $\{10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}, 1, 100\}$ on the validation year. The score is the root-mean-square error of the full standardised state, averaged over the validation lead times $\mathcal{L}_{\mathrm{val}} = \{3, 6, 12, 18, 30, 36\}$ hours:

$$
S(\lambda) = \frac{1}{|\mathcal{L}_{\mathrm{val}}|} \sum_{L \in \mathcal{L}_{\mathrm{val}}} \sqrt{\frac{1}{m\,n} \sum_{j=1}^{m} \big\| \hat{\mathbf{q}}_j(L) - \mathbf{q}_j(L) \big\|^2}. \tag{17}
$$

Lead times that are multiples of 24 hours are excluded because persistence at those leads corresponds to the same hour on the previous day, which already reproduces the diurnal cycle without any dynamics.

### H. Evaluation metrics and baselines

The forecast error of variable $v$ at lead time $L$ is reported in physical units:

$$
\mathrm{RMSE}_v(L) = \sigma_v \sqrt{\frac{1}{m N_c} \sum_{j=1}^{m} \sum_{i \in v} \big(\hat{q}_{j,i}(L) - q_{j,i}(L)\big)^2}, \tag{18}
$$

and the skill relative to a reference forecast is

$$
\mathrm{SS} = 1 - \frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{ref}}}. \tag{19}
$$

The reference forecasts are:

- **Persistence:** $\hat{\mathbf{q}}(t + L) = \mathbf{q}(t)$.
- **Hour-of-day climatology:** the training mean state for the target hour of day.
- **Projection floor:** the error of reconstructing the true state with $r$ POD modes, which is the lowest error any $r$-mode model can reach.

---

## Experimental Setup

| Item | Value |
|---|---|
| Reduced dimension $r$ | 10 |
| State dimension $n$ | 5325 |
| Time step | 1 hour |
| Maximum lead time | 168 hours |
| Evaluated lead times | 1, 3, 6, 12, 18, 24, 30, 36, 48, 72, 120, 168 hours |
| Dissipation margin $\varepsilon$ | $10^{-3}$ per hour |
| Free-run test | 720 hours (30 days) from a test start time |
| Software | Python, NumPy, SciPy, scikit-learn, xarray, GeoPandas, regionmask |

---

## Results

### A. POD energy and manifold reconstruction

| Modes | Energy captured $E(r)$ |
|---|---|
| 1 | 0.4259 |
| 3 | 0.7891 |
| 5 | 0.8756 |
| 10 | 0.9210 |
| 20 | 0.9534 |
| 40 | 0.9743 |
| 60 | 0.9834 |

Reconstruction RMSE on the test period in standardised units:

| Modes | Linear POD | Quadratic manifold | Improvement |
|---|---|---|---|
| 3 | 0.3706 | 0.3566 | 3.76 % |
| 5 | 0.2906 | 0.2738 | 5.81 % |
| 10 | 0.2306 | 0.2142 | 7.12 % |
| 20 | 0.1816 | 0.1679 | 7.55 % |
| 40 | 0.1366 | 0.1361 | 0.37 % |

![POD energy decay](figures/A_pod_energy.png)

### B. Regularisation selection on the validation year

Persistence validation score: 0.7740.

| Model | Selected $\lambda$ | Validation score | Skill vs persistence |
|---|---|---|---|
| `linear` | $10^{-4}$ | 0.5266 | 31.97 % |
| `quad` | 1 | 0.5704 | 26.30 % |
| `energy` | 1 | 0.6383 | 17.53 % |
| `linear_dis` | $10^{-2}$ | 0.4742 | 38.74 % |
| `energy_dis` | $10^{-2}$ | 0.4836 | 37.52 % |

### C. Forecast error on the test period

Two-metre temperature RMSE in degrees C:

| Lead (h) | Persistence | Hour climatology | Projection floor | `linear` | `quad` | `energy` | `linear_dis` | `energy_dis` |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.932 | 2.785 | 0.763 | 1.011 | 0.942 | 0.970 | 1.029 | 1.022 |
| 3 | 2.338 | 2.748 | 0.727 | 1.188 | 1.344 | 1.340 | 1.211 | 1.219 |
| 6 | 4.354 | 2.785 | 0.761 | 2.156 | 2.775 | 2.556 | 1.958 | 1.927 |
| 12 | 5.883 | 2.781 | 0.759 | 2.156 | 3.230 | 2.837 | 1.982 | 2.090 |
| 18 | 4.388 | 2.778 | 0.757 | 2.409 | 3.671 | 3.928 | 2.013 | 2.230 |
| 24 | 1.164 | 2.778 | 0.757 | 2.537 | 4.060 | 4.971 | 1.907 | 2.318 |
| 36 | 5.902 | 2.779 | 0.756 | 2.559 | 3.826 | 5.512 | 2.017 | 2.603 |
| 72 | 1.595 | 2.780 | 0.756 | 2.263 | 4.817 | 10.332 | 2.123 | 2.524 |
| 168 | 1.989 | 2.763 | 0.753 | 2.766 | diverged | 18.206 | 2.134 | 2.504 |

At lead times of 3, 6, 12, 18, 30 and 36 hours, the two models with a dissipative linear operator have lower temperature error than persistence and hour-of-day climatology. At 1 hour and at lead times that are multiples of 24 hours, persistence has the lowest error.

Error by variable at a lead time of 6 hours, in physical units:

| Variable | Persistence | Hour climatology | `linear` | `quad` | `energy` | `linear_dis` | `energy_dis` |
|---|---|---|---|---|---|---|---|
| t2m (degrees C) | 4.354 | 2.785 | 2.156 | 2.775 | 2.556 | 1.958 | 1.927 |
| d2m (degrees C) | 1.915 | 2.538 | 1.631 | 1.466 | 1.457 | 1.522 | 1.402 |
| u10 (m/s) | 1.390 | 2.036 | 1.067 | 1.066 | 1.095 | 1.041 | 1.001 |
| v10 (m/s) | 1.025 | 1.449 | 0.844 | 0.827 | 0.848 | 0.812 | 0.791 |
| sp (hPa) | 2.333 | 2.919 | 2.162 | 2.133 | 2.136 | 2.126 | 2.049 |

### D. Stability of the 30-day free run

The largest reduced-state magnitude in the training data is 107.5.

| Model | Finite | Max $\lvert \mathbf{z} \rvert$ | Mean energy / training energy | Max Re eigenvalue of $\mathbf{A}$ (per hour) | Max nonlinear energy term |
|---|---|---|---|---|---|
| `linear` | yes | 63.81 | 0.772 | 0.0001 | 0 |
| `quad` | yes | 200.08 | 2.951 | 0.0017 | 901.2 |
| `energy` | yes | 2047.83 | 446.779 | 0.0032 | 0 |
| `linear_dis` | yes | 56.74 | 0.847 | -0.0050 | 0 |
| `energy_dis` | yes | 52.72 | 0.718 | -0.0105 | 0 |



---

##  Conclusion

A ten-mode POD basis captures 92.1 percent of the snapshot energy of the five-variable Tamil Nadu state, and a quadratic manifold reduces the reconstruction error by up to 7.55 percent at equal dimension. In the operator inference study, a dissipative linear operator, obtained by projecting the learned operator onto the form $\mathbf{J} - \mathbf{D}$, provides stability over a 30-day free run and the best learned forecasts at lead times of 3 to 36 hours. An energy-preserving nonlinear term alone does not prevent instability. Possible extensions include fitting the model directly under the dissipativity constraint, operator inference in quadratic manifold coordinates, larger reduced dimensions, and Hamiltonian test problems in which exact energy conservation is the correct physics.


[11] P. Benner, S. Gugercin, and K. Willcox, "A survey of projection-based model order reduction methods for parametric dynamical systems," *SIAM Review*, vol. 57, no. 4, pp. 483-531, 2015.
