# Structure-preserving reduced-order modeling of regional weather over Tamil Nadu

This repository builds a reduced-order model (ROM) of hourly near-surface weather over Tamil Nadu, India, from ERA5-Land reanalysis. The full state has five variables on 1065 land cells (5325 degrees of freedom). Proper orthogonal decomposition (POD) compresses it to ten modes that keep 92.1 percent of the snapshot energy. A quadratic manifold is compared with the linear POD subspace for reconstruction.

The reduced dynamics are learned with operator inference: a quadratic ODE with known calendar forcing, fitted by regularised least squares and integrated with fourth-order Runge-Kutta. Two structural constraints are tested alone and together, an energy-preserving nonlinear term and a dissipative linear operator. On the 2024 test year, models with a dissipative linear operator stay bounded over a 30-day free run and have the lowest 2 m temperature error among the learned models at lead times of 6 to 36 hours. The model with an energy-preserving nonlinear term alone does not stay bounded.

**Keywords:** reduced-order modeling, operator inference, proper orthogonal decomposition, quadratic manifold, energy-preserving dynamics, dissipative systems, ERA5-Land, scientific machine learning

## Introduction

Numerical weather prediction solves the governing equations on fine grids at high cost. A reduced-order model replaces the full state with a few coordinates and learns how they evolve from data. Operator inference [1], [3] does this without access to the full-order solver: it fits the operators of a polynomial model to projected snapshots. Advection terms are quadratic in the state, so a quadratic model is a natural choice.

Learned models are often unstable over long runs. Structure-preserving formulations build in physical properties, such as an energy-preserving nonlinear term [5] or Hamiltonian and Lagrangian structure [6], [7]. Near-surface weather is forced and dissipative, not Hamiltonian, so total energy is not conserved here. This work tests two constraints that fit a forced dissipative system: a nonlinear term that cannot create energy and a linear operator that cannot amplify it.

The repository contains:

1. A reproducible pipeline that builds a state vector from ERA5-Land hourly fields over Tamil Nadu and compresses it with POD.
2. A comparison of the linear POD subspace and a quadratic manifold for state reconstruction.
3. An operator inference study with five model variants that separate the effect of the energy-preserving term from the effect of the dissipative operator.
4. A forecast and stability evaluation against persistence and hour-of-day climatology on a held-out test year.

## Data

### Source

ERA5-Land hourly reanalysis [9], [10] on a 0.1 degree grid. The domain spans 8.0 to 13.5 degrees N and 76.0 to 80.5 degrees E (56 by 46 grid points). A cell is kept if it is land inside the Tamil Nadu administrative boundary, which leaves $`N_c = 1065`$ cells.

### Variables

| Symbol | Variable                  | Unit      |
| ------ | ------------------------- | --------- |
| t2m    | 2 m temperature           | degrees C |
| d2m    | 2 m dew point temperature | degrees C |
| u10    | 10 m zonal wind           | m/s       |
| v10    | 10 m meridional wind      | m/s       |
| sp     | Surface pressure          | hPa       |

Precipitation is left out. It is intermittent and non-smooth, which suits neither POD nor polynomial reduced models.

### State vector

Each variable $`v`$ is standardised with its mean $`\mu_v`$ and standard deviation $`\sigma_v`$, computed over the training period only. The state at hour $`t`$ stacks the five standardised fields over the $`N_c`$ cells:

```math
\mathbf{q}(t) \in \mathbb{R}^{n}, \qquad n = 5N_c = 5325 \tag{1}
```

### Temporal split

Whole calendar years are used, so no sample shares a period across splits.

| Split      | Period        | Hours used |
| ---------- | ------------- | ---------- |
| Training   | 2020 and 2022 | 16008      |
| Validation | 2023          | 8760       |
| Test       | 2024          | 7368       |

Only hours present in the downloaded files are used, and no training or forecast window spans a missing hour. There are 730 forecast start times for validation (every 12 hours) and 1144 for testing (every 6 hours).

## Method

### Proper orthogonal decomposition

Let $`\bar{\mathbf{q}}`$ be the mean training state and $`\mathbf{X} = [\mathbf{q}(t_1)-\bar{\mathbf{q}},\ \dots,\ \mathbf{q}(t_m)-\bar{\mathbf{q}}]`$ the centred snapshot matrix. Its singular value decomposition is

```math
\mathbf{X} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{Y}^{\top} \tag{2}
```

and the POD basis $`\mathbf{V}_r \in \mathbb{R}^{n \times r}`$ holds the first $`r`$ columns of $`\mathbf{U}`$. The reduced coordinates and the linear reconstruction are

```math
\mathbf{z}(t) = \mathbf{V}_r^{\top}\big(\mathbf{q}(t) - \bar{\mathbf{q}}\big), \qquad
\hat{\mathbf{q}}_{\mathrm{lin}} = \bar{\mathbf{q}} + \mathbf{V}_r\,\mathbf{z} \tag{3}
```

The fraction of snapshot energy captured by $`r`$ modes is

```math
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^{2}}{\sum_{i=1}^{\min(n,m)} \sigma_i^{2}} \tag{4}
```

A randomised SVD is applied to every third training hour. The reduced dimension for the dynamics is $`r = 10`$.

### Quadratic manifold

A quadratic manifold [4] adds a curved correction to the linear subspace:

```math
\hat{\mathbf{q}}_{\mathrm{quad}} = \bar{\mathbf{q}} + \mathbf{V}_r\,\mathbf{z} + \mathbf{W}\,\operatorname{vech}\!\big(\mathbf{z}\mathbf{z}^{\top}\big) \tag{5}
```

Here $`\operatorname{vech}(\cdot)`$ stacks the $`r(r+1)/2`$ unique products $`z_i z_j`$ with $`i \le j`$. The matrix $`\mathbf{W}`$ is the ridge-regularised least-squares map from the quadratic features to the residual of the linear reconstruction:

```math
\mathbf{W} = \arg\min_{\mathbf{W}} \; \big\lVert \mathbf{R} - \mathbf{W}\mathbf{F} \big\rVert_F^{2} + \gamma\,\lVert \mathbf{W} \rVert_F^{2} \tag{6}
```

with $`\mathbf{R} = (\mathbf{q} - \bar{\mathbf{q}}) - \mathbf{V}_r\mathbf{z}`$ and $`\mathbf{F}`$ the matrix of scaled quadratic features.

### Reduced model by operator inference

The reduced state follows a quadratic ODE with known forcing:

```math
\dot{\mathbf{z}} = \mathbf{c} + \mathbf{A}\,\mathbf{z} + \mathbf{H}\,\operatorname{vech}\!\big(\mathbf{z}\mathbf{z}^{\top}\big) + \mathbf{B}\,\mathbf{u}(t) \tag{7}
```

with $`\mathbf{c} \in \mathbb{R}^{r}`$, $`\mathbf{A} \in \mathbb{R}^{r \times r}`$, $`\mathbf{H} \in \mathbb{R}^{r \times r(r+1)/2}`$ and $`\mathbf{B} \in \mathbb{R}^{r \times 4}`$. The forcing encodes the hour of day $`h`$ and the day of year $`d`$:

```math
\mathbf{u}(t) = \Big[\sin\tfrac{2\pi h}{24},\ \cos\tfrac{2\pi h}{24},\ \sin\tfrac{2\pi d}{366},\ \cos\tfrac{2\pi d}{366}\Big]^{\top} \tag{8}
```

The time derivative is a central difference over three consecutive hours, with $`\Delta t = 1`$ hour:

```math
\dot{\mathbf{z}}(t_k) \approx \frac{\mathbf{z}(t_{k+1}) - \mathbf{z}(t_{k-1})}{2\,\Delta t} \tag{9}
```

Collect the unknown operators in $`\boldsymbol{\Theta}`$ and the matching data features in $`\mathbf{G}`$. The operators solve the regularised least-squares problem

```math
\min_{\boldsymbol{\Theta}} \; \big\lVert \mathbf{G}\boldsymbol{\Theta} - \dot{\mathbf{Z}} \big\rVert_F^{2} + \lambda\,\lVert \boldsymbol{\Theta} \rVert_F^{2} \tag{10}
```

with column equilibration of $`\mathbf{G}`$ and no penalty on the constant term. Forecasts integrate Eq. (7) with the classical fourth-order Runge-Kutta method. Writing $`f(\mathbf{z}, \mathbf{u})`$ for the right-hand side of Eq. (7):

```math
\begin{aligned}
\mathbf{s}_1 &= f\big(\mathbf{z}_k,\ \mathbf{u}(t_k)\big), \\
\mathbf{s}_2 &= f\big(\mathbf{z}_k + \tfrac{\Delta t}{2}\mathbf{s}_1,\ \mathbf{u}(t_k + \tfrac{\Delta t}{2})\big), \\
\mathbf{s}_3 &= f\big(\mathbf{z}_k + \tfrac{\Delta t}{2}\mathbf{s}_2,\ \mathbf{u}(t_k + \tfrac{\Delta t}{2})\big), \\
\mathbf{s}_4 &= f\big(\mathbf{z}_k + \Delta t\,\mathbf{s}_3,\ \mathbf{u}(t_k + \Delta t)\big), \\
\mathbf{z}_{k+1} &= \mathbf{z}_k + \tfrac{\Delta t}{6}\big(\mathbf{s}_1 + 2\mathbf{s}_2 + 2\mathbf{s}_3 + \mathbf{s}_4\big)
\end{aligned} \tag{11}
```

The forcing is evaluated analytically at the start, middle and end of each step.

### Energy-preserving nonlinear term

Define the reduced energy $`\mathcal{E}(\mathbf{z}) = \tfrac{1}{2}\lVert\mathbf{z}\rVert^{2}`$. Because $`\mathbf{V}_r`$ has orthonormal columns, this is the energy of the reduced state. The nonlinear term changes the energy at the rate $`\mathbf{z}^{\top}\mathbf{H}\operatorname{vech}(\mathbf{z}\mathbf{z}^{\top})`$, so it is energy preserving when

```math
\sum_{i=1}^{r}\ \sum_{j \le l} H_{i,jl}\; z_i\, z_j\, z_l = 0 \quad \text{for all } \mathbf{z} \in \mathbb{R}^{r} \tag{12}
```

Eq. (12) is a cubic polynomial in $`\mathbf{z}`$ that has to vanish identically. That happens if and only if the total coefficient of every cubic monomial $`z_a z_b z_c`$ ($`a \le b \le c`$) is zero, which gives a set of linear equality constraints on the entries of $`\mathbf{H}`$. Write them as $`\mathbf{C}\operatorname{vec}(\mathbf{H}) = \mathbf{0}`$ and take an orthonormal basis $`\mathbf{N}`$ of the null space of $`\mathbf{C}`$:

```math
\operatorname{vec}(\mathbf{H}) = \mathbf{N}\,\mathbf{a} \tag{13}
```

The energy-preserving fit is then a linear least-squares problem in the coefficients $`\mathbf{a}`$. For $`r = 10`$ the unconstrained $`\mathbf{H}`$ has 550 entries, there are 220 constraints, and 330 free parameters remain. As a numerical check, $`\max\lvert\mathbf{z}^{\top}\mathbf{H}\operatorname{vech}(\mathbf{z}\mathbf{z}^{\top})\rvert`$ is of order $`10^{-14}`$ for operators built from $`\mathbf{N}`$.

### Dissipative linear operator

The linear operator is split into a skew-symmetric part and a negative definite symmetric part:

```math
\mathbf{A} = \mathbf{J} - \mathbf{D}, \qquad \mathbf{J} = -\mathbf{J}^{\top}, \qquad \mathbf{D} = \mathbf{D}^{\top} \succ 0 \tag{14}
```

Let $`\mathbf{K} = \tfrac{1}{2}(\mathbf{A} - \mathbf{A}^{\top})`$ and $`\mathbf{S} = \tfrac{1}{2}(\mathbf{A} + \mathbf{A}^{\top}) = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}`$. The nearest dissipative operator comes from clipping the eigenvalues of $`\mathbf{S}`$:

```math
\mathbf{A}_{\mathrm{dis}} = \mathbf{K} + \mathbf{Q}\,\operatorname{diag}\big(\min(\lambda_i,\, -\varepsilon)\big)\,\mathbf{Q}^{\top} \tag{15}
```

with $`\varepsilon = 10^{-3}`$ per hour. The model is first fitted with a free $`\mathbf{A}`$. Then $`\mathbf{A}`$ is replaced by $`\mathbf{A}_{\mathrm{dis}}`$, and $`\mathbf{c}`$, $`\mathbf{H}`$ and $`\mathbf{B}`$ are refitted with $`\mathbf{A}_{\mathrm{dis}}`$ held fixed.

With an energy-preserving $`\mathbf{H}`$ and a dissipative $`\mathbf{A}`$, the skew part and the nonlinear term drop out of the energy balance of Eq. (7):

```math
\frac{d}{dt}\,\tfrac{1}{2}\lVert\mathbf{z}\rVert^{2}
= -\mathbf{z}^{\top}\mathbf{D}\,\mathbf{z} + \mathbf{z}^{\top}\big(\mathbf{c} + \mathbf{B}\mathbf{u}(t)\big)
\le -\lambda_{\min}(\mathbf{D})\,\lVert\mathbf{z}\rVert^{2} + \lVert\mathbf{c} + \mathbf{B}\mathbf{u}(t)\rVert\,\lVert\mathbf{z}\rVert \tag{16}
```

The forcing is bounded, so Eq. (16) gives

```math
\limsup_{t \to \infty} \lVert\mathbf{z}(t)\rVert \le \frac{\sup_t \lVert\mathbf{c} + \mathbf{B}\mathbf{u}(t)\rVert}{\lambda_{\min}(\mathbf{D})}
```

### Model variants

| Model        | Nonlinear term H            | Linear operator A     |
| ------------ | --------------------------- | --------------------- |
| `linear`     | none                        | free                  |
| `quad`       | free                        | free                  |
| `energy`     | energy preserving, Eq. (12) | free                  |
| `linear_dis` | none                        | dissipative, Eq. (15) |
| `energy_dis` | energy preserving, Eq. (12) | dissipative, Eq. (15) |

### Regularisation selection

The weight $`\lambda`$ is chosen from $`\{10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}, 1, 100\}`$ on the validation year. The score is the root-mean-square error of the full standardised state over the $`M`$ validation forecasts, averaged over the validation lead times $`\mathcal{L}_{\mathrm{val}} = \{3, 6, 12, 18, 30, 36\}`$ hours:

```math
S(\lambda) = \frac{1}{\lvert\mathcal{L}_{\mathrm{val}}\rvert} \sum_{L \in \mathcal{L}_{\mathrm{val}}} \sqrt{\frac{1}{M\,n} \sum_{j=1}^{M} \big\lVert \hat{\mathbf{q}}_j(L) - \mathbf{q}_j(L) \big\rVert^{2}} \tag{17}
```

Lead times that are multiples of 24 hours are excluded. Persistence at those leads is the same hour of the previous day, which already reproduces the diurnal cycle without any dynamics.

### Metrics and baselines

The forecast error of variable $`v`$ at lead time $`L`$ is reported in physical units:

```math
\mathrm{RMSE}_v(L) = \sigma_v \sqrt{\frac{1}{M\,N_c} \sum_{j=1}^{M}\ \sum_{i \in v} \big(\hat{q}_{j,i}(L) - q_{j,i}(L)\big)^{2}} \tag{18}
```

Skill relative to a reference forecast is

```math
\mathrm{SS} = 1 - \frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{ref}}} \tag{19}
```

The references are:

- Persistence: $`\hat{\mathbf{q}}(t + L) = \mathbf{q}(t)`$.
- Hour-of-day climatology: the training mean state for the target hour of day.
- Projection floor: the error of reconstructing the true state with $`r`$ POD modes, which is the lowest error any $`r`$-mode model can reach.

## Experimental setup

| Item                             | Value                                                             |
| -------------------------------- | ----------------------------------------------------------------- |
| Reduced dimension $`r`$          | 10                                                                |
| State dimension $`n`$            | 5325                                                              |
| Time step                        | 1 hour                                                            |
| Maximum lead time                | 168 hours                                                         |
| Evaluated lead times             | 1, 3, 6, 12, 18, 24, 30, 36, 48, 72, 120, 168 hours               |
| Dissipation margin $`\varepsilon`$ | $`10^{-3}`$ per hour                                            |
| Free-run test                    | 720 hours (30 days) from a test start time                        |
| Software                         | Python, NumPy, SciPy, scikit-learn, xarray, GeoPandas, regionmask |

## Results

### POD energy and manifold reconstruction

| Modes | Energy captured |
| ----- | --------------- |
| 1     | 0.4259          |
| 3     | 0.7891          |
| 5     | 0.8756          |
| 10    | 0.9210          |
| 20    | 0.9534          |
| 40    | 0.9743          |
| 60    | 0.9834          |

Reconstruction RMSE on the test year, in standardised units:

| Modes | Linear POD | Quadratic manifold | Improvement |
| ----- | ---------- | ------------------ | ----------- |
| 3     | 0.3706     | 0.3566             | 3.76 %      |
| 5     | 0.2906     | 0.2738             | 5.81 %      |
| 10    | 0.2306     | 0.2142             | 7.12 %      |
| 20    | 0.1816     | 0.1679             | 7.55 %      |
| 40    | 0.1366     | 0.1361             | 0.37 %      |

### Regularisation selection on the validation year

Persistence scores 0.7740 on the validation year.

| Model        | Selected $`\lambda`$ | Validation score | Skill vs persistence |
| ------------ | -------------------- | ---------------- | -------------------- |
| `linear`     | $`10^{-4}`$          | 0.5266           | 31.97 %              |
| `quad`       | 1                    | 0.5704           | 26.30 %              |
| `energy`     | 1                    | 0.6383           | 17.53 %              |
| `linear_dis` | $`10^{-2}`$          | 0.4742           | 38.74 %              |
| `energy_dis` | $`10^{-2}`$          | 0.4836           | 37.52 %              |

### Forecast error on the test year

Two-metre temperature RMSE in degrees C:

| Lead (h) | Persistence | Hour climatology | Projection floor | `linear` | `quad`   | `energy` | `linear_dis` | `energy_dis` |
| -------- | ----------- | ---------------- | ---------------- | -------- | -------- | -------- | ------------ | ------------ |
| 1        | 0.932       | 2.785            | 0.763            | 1.011    | 0.942    | 0.970    | 1.029        | 1.022        |
| 3        | 2.338       | 2.748            | 0.727            | 1.188    | 1.344    | 1.340    | 1.211        | 1.219        |
| 6        | 4.354       | 2.785            | 0.761            | 2.156    | 2.775    | 2.556    | 1.958        | 1.927        |
| 12       | 5.883       | 2.781            | 0.759            | 2.156    | 3.230    | 2.837    | 1.982        | 2.090        |
| 18       | 4.388       | 2.778            | 0.757            | 2.409    | 3.671    | 3.928    | 2.013        | 2.230        |
| 24       | 1.164       | 2.778            | 0.757            | 2.537    | 4.060    | 4.971    | 1.907        | 2.318        |
| 36       | 5.902       | 2.779            | 0.756            | 2.559    | 3.826    | 5.512    | 2.017        | 2.603        |
| 72       | 1.595       | 2.780            | 0.756            | 2.263    | 4.817    | 10.332   | 2.123        | 2.524        |
| 168      | 1.989       | 2.763            | 0.753            | 2.766    | diverged | 18.206   | 2.134        | 2.504        |

At 3, 6, 12, 18, 30 and 36 hours, both dissipative models beat persistence and hour-of-day climatology. Among the learned models, a dissipative one has the lowest error at every evaluated lead from 6 to 36 hours. At 3 hours `linear` is slightly ahead (1.188 against 1.211 and 1.219). At 1 hour and at multiples of 24 hours, persistence has the lowest error.

Error by variable at a 6-hour lead, in physical units:

| Variable        | Persistence | Hour climatology | `linear` | `quad` | `energy` | `linear_dis` | `energy_dis` |
| --------------- | ----------- | ---------------- | -------- | ------ | -------- | ------------ | ------------ |
| t2m (degrees C) | 4.354       | 2.785            | 2.156    | 2.775  | 2.556    | 1.958        | 1.927        |
| d2m (degrees C) | 1.915       | 2.538            | 1.631    | 1.466  | 1.457    | 1.522        | 1.402        |
| u10 (m/s)       | 1.390       | 2.036            | 1.067    | 1.066  | 1.095    | 1.041        | 1.001        |
| v10 (m/s)       | 1.025       | 1.449            | 0.844    | 0.827  | 0.848    | 0.812        | 0.791        |
| sp (hPa)        | 2.333       | 2.919            | 2.162    | 2.133  | 2.136    | 2.126        | 2.049        |

### Stability of the 30-day free run

The largest reduced-state magnitude in the training data is 107.5.

| Model        | Finite | Max norm of z | Mean energy / training energy | Max Re eigenvalue of A (per hour) | Max nonlinear energy term |
| ------------ | ------ | ------------- | ----------------------------- | --------------------------------- | ------------------------- |
| `linear`     | yes    | 63.81         | 0.772                         | 0.0001                            | 0                         |
| `quad`       | yes    | 200.08        | 2.951                         | 0.0017                            | 901.2                     |
| `energy`     | yes    | 2047.83       | 446.779                       | 0.0032                            | 0                         |
| `linear_dis` | yes    | 56.74         | 0.847                         | -0.0050                           | 0                         |
| `energy_dis` | yes    | 52.72         | 0.718                         | -0.0105                           | 0                         |

The `energy` model reaches about 19 times the largest training state. The dissipative models stay below it, and so does the unconstrained `linear` model, so the clearest contrast is between `energy` and the two dissipative variants.

## Conclusions

A ten-mode POD basis captures 92.1 percent of the snapshot energy of the Tamil Nadu state, and a quadratic manifold lowers the reconstruction error by up to 7.55 percent at the same dimension. In the operator inference study, projecting the learned linear operator onto the form $`\mathbf{J} - \mathbf{D}`$ gives a stable 30-day free run and the best learned forecasts from 6 to 36 hours. An energy-preserving nonlinear term alone does not prevent instability.

## Limits and next steps

- The 30-day free run starts from one test start time, so the stability result rests on a single trajectory.
- Only one test year (2024) is used.
- Persistence beats every model at 1 hour and at multiples of 24 hours.
- Next steps: fit the model directly under the dissipativity constraint, run operator inference in quadratic-manifold coordinates, try larger reduced dimensions, and test on Hamiltonian problems where exact energy conservation is the right physics.

## Reproducing the results

Data are not included. Download ERA5-Land hourly fields for the five variables, the domain and the years listed above from the Copernicus Climate Data Store, then point the notebook at your files. Install the dependencies and run `tamilnadu_reduced_order_modeling_v2.ipynb` from top to bottom:

```
pip install -r requirements.txt
```

## References

<!-- Add entries [1] to [10] to match the citations in the text. -->

[11] P. Benner, S. Gugercin, and K. Willcox, "A survey of projection-based model order reduction methods for parametric dynamical systems," *SIAM Review*, vol. 57, no. 4, pp. 483-531, 2015.
