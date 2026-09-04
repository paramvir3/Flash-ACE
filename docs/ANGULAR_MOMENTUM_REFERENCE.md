# Angular-momentum reference for TRACE

Distilled from D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii,
*Quantum Theory of Angular Momentum* (World Scientific, 1988; Open Access
2021, CC BY 4.0, doi:10.1142/0270). Cited below as **VMK**.

Referencing convention follows the book: `VMK 3.1(27)` = Chapter 3, Section 1,
Equation (27). Printed page `p` is PDF page `p + 13`.

**Purpose.** VMK is a handbook, not a textbook -- the Preface states that
"most of the formulas and relationships are given without proof", and roughly
150 of its 528 pages are numerical/algebraic lookup tables (Secs. 4.21,
8.12-8.13, 9.11-9.12, 10.11-10.13) that `e3nn` computes at runtime. This file
records the parts that are *load-bearing for TRACE*: definitions, phase and
parity conventions, symmetry relations, recoupling identities, and sum rules --
together with what each one implies for the code.

**Status.** Chapter 3 complete. Chapter 5 complete for Secs. 5.1-5.12, 5.16-5.17.
See [Reading plan](#reading-plan) for the rest.

---

## Contents

- [Ch. 3 -- Irreducible tensors](#ch-3--irreducible-tensors) *(complete)*
- [Ch. 5 -- Spherical harmonics](#ch-5--spherical-harmonics) *(5.1-5.12, 5.16-5.17)*
- [Convention bridge: VMK vs e3nn](#convention-bridge-vmk-vs-e3nn) *(numerically verified)*
- [TRACE implications so far](#trace-implications-so-far)
- [Reading plan](#reading-plan)

---

## Ch. 3 -- Irreducible tensors

*(VMK pp. 61-71; PDF 74-84)*

### 3.1.1 Definition and phase convention

An irreducible tensor `M_J` of rank `J` is a set of `2J+1` components
`M_JM`, `M = -J..J`, obeying

```
[J_±1, M_JM] = ∓(1/√2) e^{±iδ} √(J(J+1) − M(M±1)) · M_{J,M±1}
[J_0 , M_JM] = M · M_JM                                              VMK 3.1(1)

[J_μ , M_JM] = e^{iMδ} √(J(J+1)) · C^{J,M+μ}_{J M, 1 μ} · M_{J,M+μ}  VMK 3.1(2)
[J²  , M_JM] = J(J+1) · M_JM                                         VMK 3.1(3)
```

The book adopts `δ = 0` and the positive square-root branch. The overall phase
for integer `J` is fixed by

```
(M_JM)* = (−1)^{−M} M_{J,−M}                                         VMK 3.1(4)
```

which **coincides with the spherical-harmonic convention of Ch. 5**. An
alternative tensor `M̃_J = i^J M_J` (VMK 3.1(5)) satisfies
`(M̃_JM)* = (−1)^{J−M} M̃_{J,−M}` (VMK 3.1(6)) and is the one for which
Hermitian tensor operators behave simply.

> **Convention check for TRACE.** `e3nn` uses *real* spherical harmonics in
> `"component"` normalization, not VMK's complex `Y_lm`. Every phase factor
> below is stated in VMK's complex convention. Before transcribing any formula
> into code, confirm the real-basis image of the phase. This is the single
> most common source of silent sign errors in equivariant NN code.

### 3.1.3 Behaviour under rotation

```
M_JM'(X') = D̂(α,β,γ) M_JM'(X) D̂(α,β,γ)^{-1}
          = Σ_M M_JM(X) · D^J_{MM'}(α,β,γ)                          VMK 3.1(11)
```

### 3.1.4 Behaviour under inversion -- parity

An integer-rank tensor splits into definite-parity parts
`M_J = M_J^{(+1)} + M_J^{(−1)}` (VMK 3.1(12)) with

```
P̂_r M_J^{(π_J)} P̂_r^{-1} = π_J M_J^{(π_J)},   π_J = ±1              VMK 3.1(13)
```

- `π_J = (−1)^J`   -> **true / polar** tensor of rank `J`
- `π_J = (−1)^{J+1}` -> **pseudotensor / axial** tensor of rank `J`

> **TRACE.** The manuscript's "natural parity" truncation `(l, (−1)^l)` keeps
> exactly the *true/polar* tensors and discards every pseudotensor. VMK's
> terminology is the right one to use in the paper: TRACE retains polar
> irreducible tensors and omits axial ones. This is a stronger and clearer
> statement than "natural parity", and it makes the physical content explicit
> -- the model cannot represent a parity-odd scalar (a pseudoscalar), which is
> exactly why it assigns equal energy to enantiomers.

### 3.1.7 Irreducible tensor product -- the ACE coupling

```
L_JM = Σ_{M1 M2} C^{JM}_{J1 M1, J2 M2} · M_{J1M1} · N_{J2M2}         VMK 3.1(20)
L_J  ≡ {M_J1 ⊗ N_J2}_J                                               VMK 3.1(21)
```

This **is** manuscript Eq. (13) with the learned channel mixing `W` stripped
out. The inverse (decomposition of the direct product) is

```
M_{J1M1} N_{J2M2} = Σ_{J=|J1−J2|}^{J1+J2} C^{JM}_{J1M1 J2M2} L_JM    VMK 3.1(22)
```

#### The two structural identities

```
{M_J1 ⊗ N_J2}_JM = (−1)^{J1+J2−J} {N_J2 ⊗ M_J1}_JM   (commuting)     VMK 3.1(27)

{M_J ⊗ M_J}_I = 0   for  I = 2J−1, 2J−3, …           (self-product)  VMK 3.1(29)
```

For non-commuting tensors VMK 3.1(28) adds a commutator term
`R^{J1J2}_JM` defined by VMK 3.1(25)-(26). Our features commute (they are
numbers, not operators), so **VMK 3.1(27) applies exactly**.

> **TRACE -- verified consequence.** `ACEV2Descriptor.forward` computes the
> first correlation as `contractions[0](density, density)`, i.e. `TP(A, A)`
> with the *same* tensor in both slots. VMK 3.1(27) then makes a large block of
> the learned weights structurally unidentifiable. Measured on the production
> configuration (`l_max=2`, `correlation_channels=16`,
> `irreps_correlation = 16x0e+8x1o+4x2e`):
>
> | source of redundancy | dead weights |
> |---|---|
> | antisymmetric part of `W[c1,c2,c]` on all `l1 == l2` paths | 2,600 |
> | mirror pairs `(l1,l2)->l` vs `(l2,l1)->l`, verified collinear to 6e-16 | 1,536 |
> | **total** | **4,136 of 8,768 = 47.2%** |
>
> That is **3.1% of the whole 132,005-parameter model** that cannot affect the
> output. Confirmed empirically: a large purely-antisymmetric perturbation of
> `W` changes `TP(A,A)` by 2e-14.
>
> *Why the surviving `l1 == l2` paths are all symmetric:* the antisymmetric
> couplings need `l1+l2−l` odd, which for `l1 == l2` forces odd `l`; combined
> with even total parity this lands on unnatural parity and is dropped by the
> truncation. So natural-parity truncation and VMK 3.1(29) remove the same
> channels, for the same reason.
>
> *Fixes worth trying:* (a) parameterize only the symmetric part of `W` on
> `l1 == l2` paths and merge mirror pairs -- frees ~4k parameters at identical
> expressivity; (b) use `o3.TensorSquare`, which builds the symmetrized product
> natively; (c) do not contract `A` with itself -- e.g. contract `A` against a
> *different* learned linear image of `A`, which restores the antisymmetric
> paths as genuine degrees of freedom.

### 3.1.8 Scalar products

```
(M_J · N_J) = Σ_M (−1)^M M_JM N_{J,−M}                               VMK 3.1(30)
{M_J ⊗ N_J}_00 = (1/√(2J+1)) Σ_M (−1)^{J−M} M_JM N_{J,−M}            VMK 3.1(33)
(M_J · N_J) = (−1)^{−J} √(2J+1) · {M_J ⊗ N_J}_00                     VMK 3.1(35)
```

> **TRACE.** The FFN invariants `n_icl = Σ_m |h_icm|²` (manuscript Eq. 24) are
> scalar products in the sense of VMK 3.1(30), i.e. rank-0 tensor products
> rescaled by `√(2l+1)` per VMK 3.1(35). The `√(2l+1)` factor is **`l`-dependent**,
> so the invariants fed to `scalar_ffn` carry an `l`-dependent natural scale
> (`√3` for `l=1`, `√5` for `l=2`). The `LayerNorm` in front of the FFN absorbs
> this at inference, but it biases initialization and early optimization.
> Dividing `n_icl` by `(2l+1)` before the norm would make the channels
> commensurate.

---

### 3.2 Relation to ordinary vector/tensor algebra

*(VMK pp. 65-69)*

#### 3.2.1 Vectors

`A^μ_1 = (−1)^μ A_{1,−μ}`; a polar vector is a true rank-1 tensor, an axial
vector a rank-1 pseudotensor.

```
{A_1 ⊗ B_1}_00 = −(1/√3) (A·B)                                       VMK 3.2(2)
(A_1 · B_1)    = (A·B)                                               VMK 3.2(3)
{A_1 ⊗ B_1}_1  = (i/√2) [A × B]                                      VMK 3.2(4)
```

Rank-2 components explicitly (VMK 3.2(7)):

```
{A⊗B}_{2,±2} = A_{±1}B_{±1}
{A⊗B}_{2,±1} = (1/√2)(A_{±1}B_0 + A_0 B_{±1})
{A⊗B}_{2, 0} = (1/√6)(A_{+1}B_{−1} + 2A_0B_0 + A_{−1}B_{+1})
```

Triple products (VMK 3.2(8)-(11)):

```
{{A⊗B}_0 ⊗ C}_1 = −(1/√3)(A·B) C
{{A⊗B}_1 ⊗ C}_0 = −(i/√3)[A×B]·C
{{A⊗B}_1 ⊗ C}_1 = −(1/2)[[A×B]×C] = (1/2)B(A·C) − (1/2)A(B·C)
{{A⊗B}_2 ⊗ C}_1 = √(3/5){ (1/3)C(A·B) − (1/2)B(A·C) − (1/2)A(B·C) }
```

Four-vector products are VMK 3.2(12)-(20); the useful invariants are

```
{{A⊗B}_0 ⊗ {C⊗D}_0}_0 = (1/3)(A·B)(D·C)                              VMK 3.2(12)
{{A⊗B}_1 ⊗ {C⊗D}_1}_0 = (1/(2√3)){(A·C)(B·D) − (A·D)(B·C)}           VMK 3.2(15)
{{A⊗B}_2 ⊗ {C⊗D}_2}_0 = (1/√5){ (1/2)(A·C)(B·D) − (1/3)(A·B)(C·D)
                                 + (1/2)(A·D)(B·C) }                 VMK 3.2(19)
```

#### Repeated products of one vector collapse to a single harmonic

```
{…{{A_1⊗A_1}_{l2} ⊗ A_1}_{l3} … ⊗ A_1}_{l_n m_n}
   = √(4π/(2l_n+1)) |A|^n Y_{l_n m_n}(θ,φ) · Π_{i=2}^{n} C^{l_i 0}_{1 0, l_{i−1} 0}
                                                                     VMK 3.2(22)

{…{{A_1⊗A_1}_2 ⊗ A_1}_3 … ⊗ A_1}_{nm}
   = √(4π n!/(2n+1)!!) |A|^n Y_{nm}(θ,φ)                             VMK 3.2(23)
```

> **TRACE -- this is the precise form of the "repeated-index" caveat.**
> Manuscript §II E says each `C^[q]` "also contains lower-body terms and is not
> a pure `(q+1)`-body contribution". VMK 3.2(22)-(23) makes that exact: any
> chain of tensor products of a *single* vector is one spherical harmonic times
> a power of the length. Therefore, **for an environment with one neighbour the
> entire correlation hierarchy `C^[1] … C^[q_max]` carries no more information
> than `{|r|, Y_lm(r̂)}`** -- increasing `correlation_order` buys literally
> nothing. Expressive gain from higher `q` requires the density `A_i` to sum
> over ≥ 2 *distinct* neighbours, and the gain is carried entirely by the
> cross terms.
>
> This is a testable prediction and a cheap, high-value ablation for the paper:
> plot accuracy vs `correlation_order` stratified by mean neighbour count. It
> also predicts that `q_max` should matter *less* in the dilute molecular case
> (methyl migration) than in dense CsPbI3.

#### 3.2.2 Cartesian tensors

`T_ik = E δ_ik + A_ik + S_ik` (VMK 3.2(24)) with `E = (1/3)Tr T`,
`A_ik = (1/2)(T_ik − T_ki)`, `S_ik = (1/2)(T_ik + T_ki − (2/3)δ_ik Σ_l T_ll)`.
Irreducible images: `T_00 = E` (VMK 3.2(28)); the antisymmetric part maps to an
axial vector `A_ik = ε_ikl U_l` (VMK 3.2(29)-(30)); the symmetric traceless part
maps to a rank-2 tensor (VMK 3.2(31)):

```
T_20   = S_zz
T_2±1  = ∓√(2/3)(S_xz ± i S_yz)
T_2±2  = √(1/6)(S_xx − S_yy ± 2i S_xy)
```

> **TRACE.** This is the dictionary between the Cauchy stress
> (a symmetric Cartesian rank-2 tensor) and irreducible `0e ⊕ 2e`. Useful if
> you ever want to predict or regularize stress in the irreducible basis
> instead of Voigt: the pressure is the `l=0` part and the deviatoric stress is
> the `l=2` part, and the two have different natural scales.

#### 3.2.3 Differential operators as tensor products

```
grad Φ    = {∇_1 ⊗ Φ}_1                                              VMK 3.2(32)
div A     = −√3 {∇_1 ⊗ A_1}_0                                        VMK 3.2(33)
curl A    = −i√2 {∇_1 ⊗ A_1}_1                                       VMK 3.2(34)
Δ = ∇²    = −√3 {∇_1 ⊗ ∇_1}_0                                        VMK 3.2(35)
curl grad Φ = 0  ,  div curl A = 0                        VMK 3.2(39),(40)

∇_μ = √(4π/3) ( Y_1μ ∂/∂r − (√2/r){Y_1 ⊗ L_1}_1μ )                   VMK 3.2(42)
```

Taylor expansion of a scalar field (VMK 3.2(43)-(44)):

```
Φ(r+δr) = Σ_n (1/n!) (δr)^n (u·∇)^n Φ(r)
(u·∇)^n = (−1)^n Σ_{l2…ln} (−1)^{l_n} ( {…{u_1⊗u_1}_{l2}…⊗u_1}_{l_n}
                                        · {…{∇_1⊗∇_1}_{l2}…⊗∇_1}_{l_n} )
```

> **TRACE.** VMK 3.2(44) is an *irreducible* Taylor expansion. It gives an
> analytic, equivariant expression for the local-linearization (Sobolev) term
> of manuscript Eq. (41): the `n=2` term is exactly the Hessian contraction
> that `L_Sob` is trying to penalize, decomposed into `l = 0` and `l = 2`
> pieces. Replacing the two-forward-pass finite difference with this closed
> form would remove the dropout-noise contamination identified in the code
> review (`docs/PEER_REVIEW.md` §B4) and make the regularizer exact rather than
> stochastic.

---

### 3.3 Recoupling in irreducible tensor products

*(VMK pp. 69-71)*  Notation: `Π_{abc…d} = [(2a+1)(2b+1)(2c+1)…(2d+1)]^{1/2}`.

#### 3.3.1 Valid for commuting *and* non-commuting tensors

```
{{P_a⊗Q_b}_c ⊗ R_d}_f
   = (−1)^{a+b+f+d} Σ_h Π_{hc} {a b c; d f h}_6j · {P_a ⊗ {Q_b⊗R_d}_h}_f
                                                                     VMK 3.3(1)

({P_a⊗Q_b}_c · R_c) = (−1)^{−c+a} (Π_c/Π_a) (P_a · {Q_b⊗R_c}_a)      VMK 3.3(2)
```

#### 3.3.2 Commuting tensors

```
{{P_a⊗Q_b}_c ⊗ R_d}_f
   = (−1)^{c+d+f} Σ_h Π_{ch} {a b c; f d h}_6j · {Q_b ⊗ {P_a⊗R_d}_h}_f
                                                                     VMK 3.3(8)

{{P_a⊗Q_b}_c ⊗ {R_d⊗S_e}_f}_k
   = Σ_{gh} Π_{cfgh} {a b c; d e f; g h k}_9j · {{P_a⊗R_d}_g ⊗ {Q_b⊗S_e}_h}_k
                                                                     VMK 3.3(11)

({P_a⊗Q_b}_b · Q_b) = 0   for a = 2b−1, 2b−2, …                      VMK 3.3(15)
```

#### 3.3.3 Non-commuting tensors

VMK 3.3(16)-(23) reproduce the above with additive commutator terms built from
`R^{ab}_q` (VMK 3.1(26)). **Not needed for TRACE** -- our tensors commute.

> **TRACE -- the association order of the ACE recursion is a change of basis,
> not a change of expressivity.** The code left-associates:
> `C^[q+1] = TP(C^[q], A)`, i.e. `{{{A⊗A}⊗A}⊗A}`. VMK 3.3(1) says the
> right-associated product `{A⊗{A⊗{A⊗A}}}` spans the *same* space, related by
> a single 6j symbol. VMK 3.3(11) says the balanced/pairwise scheme
> `{{A⊗A}⊗{A⊗A}}` -- which is what a cumulant-style 4-body coupling looks like
> -- is related to the left-associated one by a 9j symbol.
>
> Two consequences worth stating carefully in the manuscript:
>
> 1. **At fixed `l_max` and with fully-connected channel mixing, TRACE-v2's
>    recursion and any other association order reach the same tensor space.**
>    Any claimed advantage of a different coupling scheme (including the v4
>    "cumulant" variant) is an *optimization / conditioning* claim, not an
>    expressivity claim. Do not let the paper imply otherwise -- a referee who
>    knows VMK 3.3 will catch it immediately.
> 2. Conversely, this is a *defense*: §II E's honesty about not enumerating a
>    complete `U`-matrix basis is fine, because the recoupling identities show
>    the learned projections at each order are related to the canonical ones by
>    fixed 6j/9j transforms. The truncation is in the *channel count*, not in
>    the coupling scheme.
>
> **Concrete use:** VMK 3.3(11) gives an exact test. Build `{{A⊗A}⊗{A⊗A}}_k`
> and the left-associated `{{{A⊗A}⊗A}⊗A}_k` with unconstrained channel weights
> and verify numerically that one is in the span of the other. If it is, the v4
> ablation must be reported as conditioning, not capacity.

---

## Ch. 5 -- Spherical harmonics

*(VMK pp. 130-169; PDF 143-182. Secs. 5.13-5.15 -- zeros, extrema, special values -- skipped.)*

### 5.1 Definition, normalization, phase

```
L² Y_lm = l(l+1) Y_lm ,   L_z Y_lm = m Y_lm                          VMK 5.1(3)
∫ Y*_lm Y_l'm' dΩ = δ_ll' δ_mm'                                      VMK 5.1(6)
Y_l0(0,0) = √((2l+1)/4π)                                             VMK 5.1(10)
Y*_lm(θ,φ) = Y_lm(θ,−φ) = (−1)^m Y_{l,−m}(θ,φ)                       VMK 5.1(11)
```

The **`C`-normalization** is often more convenient for products:

```
C_lm(θ,φ) = √(4π/(2l+1)) Y_lm(θ,φ)                                   VMK 5.1(7)
C_l0(θ,φ) = P_l(cos θ) ,  C_lm(0,0) = δ_m0                           VMK 5.1(8)
```

VMK explicitly remarks after 5.6(17) that `C_lm` is the natural basis for
expanding *products* of harmonics, because the `√(4π/(2l+1))` factors cancel.

### 5.2 Explicit form and the solid-harmonic identity

```
Y_lm(θ,φ) = e^{imφ} √((2l+1)/4π · (l−m)!/(l+m)!) · P_l^m(cos θ)      VMK 5.2(1)
```

`r^l Y_lm` is a homogeneous harmonic polynomial of degree `l` (VMK 5.1(16)),
and

```
Y_lm(θ,φ) = (1/r^l) √((2l+1)!!/(4π l!)) {…{{r ⊗ r}_2 ⊗ r}_3 … ⊗ r}_lm
                                                                     VMK 5.2(40)
```

### 5.4-5.5 Symmetry and coordinate transformations

```
Y_lm(π−θ, π+φ) = (−1)^l Y_lm(θ,φ)                                    VMK 5.4(6)
D̂(α,β,γ) Y_lm = Σ_m' Y_lm' D^l_{m'm}                                 VMK 5.5(1)
P̂_r Y_lm(θ,φ) = Y_lm(π−θ, φ+π) = (−1)^l Y_lm(θ,φ)                    VMK 5.5(2)
```

The last line is the origin of TRACE's parity assignment `(l, (−1)^l)`.

### 5.6 Expansions -- the key product identities

Completeness and expansion:

```
Σ_lm Y*_lm(Ω) Y_lm(Ω') = δ(φ−φ') δ(cos θ − cos θ')                   VMK 5.6(1)
f(Ω) = Σ_lm a_lm Y_lm(Ω) ,  a_lm = ∫ Y*_lm f dΩ                      VMK 5.6(4),(5)
```

**Clebsch-Gordan series** -- product of two harmonics of the *same* direction:

```
Y_l1m1(Ω) Y_l2m2(Ω)
  = Σ_LM √((2l1+1)(2l2+1)/(4π(2L+1))) C^{L0}_{l1 0 l2 0} C^{LM}_{l1m1 l2m2} Y_LM(Ω)
                                                                     VMK 5.6(9)
```

and in irreducible-product form -- **the single most useful identity for ACE**:

```
{Y_l1(Ω) ⊗ Y_l2(Ω)}_LM
  = √((2l1+1)(2l2+1)/(4π(2L+1))) · C^{L0}_{l1 0 l2 0} · Y_LM(Ω)      VMK 5.6(14)
```

Iterated (all `l_i = 1`, `L_k = k`):

```
{…{{Y_1 ⊗ Y_1}_2 ⊗ Y_1}_3 … ⊗ Y_1}_nm = √(3^n n! / ((4π)^{n−1}(2n+1)!!)) Y_nm
                                                                     VMK 5.6(17)
```

> **Two consequences.** (i) The CG product of two harmonics of *one* direction
> is a *single* harmonic times a constant -- it carries no new angular
> information. (ii) `C^{L0}_{l1 0 l2 0} = 0` unless `l1+l2+L` is even, so **half
> the couplings vanish identically**. Both verified numerically below.

### 5.7-5.8 Recursion and differential relations

```
cos θ · Y_lm = √(((l−m+1)(l+m+1))/((2l+1)(2l+3))) Y_{l+1,m}
             + √(((l−m)(l+m))/((2l−1)(2l+1)))     Y_{l−1,m}          VMK 5.7(2)

∂Y_lm/∂φ = im Y_lm                                                   VMK 5.8(4)
L_μ Y_lm = √(l(l+1)) C^{l,m+μ}_{l m 1 μ} Y_{l,m+μ}                   VMK 5.8(2)

∇[f(r) Y_lm] = −√((l+1)/(2l+1)) (df/dr − (l/r) f) Y^{l+1}_lm
             + √(l/(2l+1))     (df/dr + ((l+1)/r) f) Y^{l−1}_lm      VMK 5.8(9)
```

> **TRACE.** VMK 5.8(9) is the exact irreducible form of the gradient of a
> radial-times-angular edge feature -- i.e. the analytic force contribution of
> one edge, decomposed into `l±1` vector spherical harmonics. This is the route
> to an analytic (non-autograd) force kernel for the LAMMPS/AOT path, and to
> checking the autograd forces against a closed form.

### 5.9 Integrals -- the Gaunt coefficient

```
∫ Y_l1m1 Y_l2m2 Y*_l3m3 dΩ
   = √((2l1+1)(2l2+1)/(4π(2l3+1))) C^{l3 0}_{l1 0 l2 0} C^{l3 m3}_{l1 m1 l2 m2}
                                                                     VMK 5.9(4)
∫ Y_l1m1 Y_l2m2 Y_l3m3 dΩ
   = √((2l1+1)(2l2+1)(2l3+1)/4π) · (3j: l1 l2 l3; 0 0 0)(3j: l1 l2 l3; m1 m2 m3)
                                                                     VMK 5.9(5)
```

### 5.10 Sums over m

```
Σ_m |Y_lm(θ,φ)|² = (2l+1)/4π                                         VMK 5.10(1)
Σ_m m |Y_lm|²    = 0                                                 VMK 5.10(2)
```

### 5.16 Bipolar and tripolar harmonics -- the two-neighbour basis

```
{Y_l1(Ω1) ⊗ Y_l2(Ω2)}_LM = Σ_{m1m2} C^{LM}_{l1m1 l2m2} Y_l1m1(Ω1) Y_l2m2(Ω2)
                                                                     VMK 5.16(1)
```

These form a **complete orthonormal set** on the pair of directions
(VMK 5.16(2),(3)), transform as a rank-`L` tensor (VMK 5.16(5)) and carry
parity `(−1)^{l1+l2}` (VMK 5.16(6)). Useful norms and reductions:

```
Σ_LM |{Y_l1(Ω1) ⊗ Y_l2(Ω2)}_LM|² = (2l1+1)(2l2+1)/(4π)²              VMK 5.16(4)
{Y_l1(Ω1) ⊗ Y_l2(Ω2)}_00 = ((−1)^{l1}/√(2l1+1)) (Y_l1(Ω1)·Y_l2(Ω2)) δ_{l1l2}
                                                                     VMK 5.16(9)
```

Tripolar harmonics (three directions) are VMK 5.16(11); their different
coupling schemes are related by the Ch. 3.3 recoupling identities.

### 5.17 Functions of two vectors -- the addition theorem

```
(Y_l(Ω1) · Y_l(Ω2)) = Σ_m Y*_lm(Ω1) Y_lm(Ω2) = ((2l+1)/4π) P_l(cos ω12)
                                                                     VMK 5.17(9)
```

Expansions of scalar functions of the two vectors:

```
(r1·r2)^n = 4π r1^n r2^n Σ_l  n!/((n−l)!!(n+l+1)!!) (Y_l(Ω1)·Y_l(Ω2))
                                       l = n, n−2, …                 VMK 5.17(13)
e^{i(r1·r2)} = 4π Σ_l i^l j_l(r1 r2) (Y_l(Ω1)·Y_l(Ω2))               VMK 5.17(14)
1/|r1−r2| = (4π/r2) Σ_l (1/(2l+1)) (r1/r2)^l (Y_l(Ω1)·Y_l(Ω2))  (r1<r2)
                                                                     VMK 5.17(21)
```

> **TRACE.** VMK 5.17(21) is the multipole expansion. If long-range
> electrostatics is ever added (Scope & Limitations names this as absent), this
> is the irreducible form that composes directly with the existing `l`-channels
> rather than requiring a separate Ewald machinery.

---

## Convention bridge: VMK vs e3nn

All three relations below were checked numerically against
`e3nn.o3.spherical_harmonics(..., normalize=True, normalization="component")`
over 2000 random directions, `l ≤ 4`.

| VMK statement | e3nn image | verified |
|---|---|---|
| `Σ_m |Y^VMK_lm|² = (2l+1)/4π` (5.10(1)) | `Σ_m |Y^e3nn_lm|² = 2l+1`, i.e. **`Y^e3nn = √(4π) · Y^VMK`** | exact, all `l ≤ 4` |
| addition theorem (5.17(9)) | `Σ_m Y^e3nn_lm(n1) Y^e3nn_lm(n2) = (2l+1) P_l(n1·n2)` | max err 1.8e-14 |
| `{Y_l1(n) ⊗ Y_l2(n)}_L = k·Y_L(n)` (5.6(14)) | holds in the real basis; `= 0` **exactly** when `l1+l2+L` odd | resid ≤ 1.2e-15 |

Measured constants `k` for `{Y_l1 ⊗ Y_l2}_L` (unnormalized CG, real basis):

| `l1,l2 → L` | 0,0→0 | 0,1→1 | 0,2→2 | 1,1→0 | 1,1→2 | 1,2→1 | 1,2→3 | 2,2→0 | 2,2→2 | 2,2→4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `k` | 1.0000 | 0.5774 | 0.4472 | 1.7321 | 0.4899 | 0.8165 | 0.4286 | 2.2361 | 0.5345 | 0.3984 |

Every `l1+l2+L` odd combination (`1,1→1`; `1,2→2`; `2,2→1`; `2,2→3`) evaluated
to **identically zero**.

> **Interpretive point for the manuscript.** TRACE's "natural parity"
> truncation `(l,(−1)^l)` is *exactly* the Gaunt selection rule
> `C^{L0}_{l1 0 l2 0} = 0` unless `l1+l2+L` even. The truncation is therefore
> not an arbitrary economy -- it keeps precisely the couplings that survive for
> harmonics of a single direction. This is a much stronger justification than
> the current text gives, and worth one sentence in §II A.

---

## TRACE implications so far

Ranked by actionability.

| # | Finding | Source | Status |
|---|---|---|---|
| 1 | 47.2% of `contractions[0]` weights (4,136 of 8,768; 3.1% of the model) are structurally unidentifiable because the first ACE contraction is `TP(A, A)` | VMK 3.1(27) | **verified numerically** |
| 2 | Association order of the correlation recursion is a 6j/9j basis change, not extra expressivity -- constrains what v4 may claim | VMK 3.3(1),(11) | derived; numerical test specified |
| 3 | For a single-neighbour environment the whole correlation hierarchy collapses to one `Y_lm` -- gain from `q_max` requires ≥2 distinct neighbours | VMK 3.2(22),(23) | derived; ablation specified |
| 4 | The Sobolev term has an exact irreducible closed form, removing the dropout-noise problem of `PEER_REVIEW.md` §B4 | VMK 3.2(44) | derived |
| 5 | FFN invariants `n_icl` carry an `l`-dependent `√(2l+1)` scale; dividing by `(2l+1)` makes channels commensurate | VMK 3.1(35) | derived |
| 6 | "Natural parity" is better described as *retaining polar tensors and discarding pseudotensors* | VMK 3.1(13) | terminology |
| 7 | Cartesian↔irreducible stress dictionary (`0e ⊕ 2e`) for irreducible-basis stress handling | VMK 3.2(24)-(31) | reference |
| **8** | **41% of the model's parameters (54,656) are structurally unidentifiable** because the species embedding has rank ≤ `n_species` | VMK 5.6(14) factorization | **verified numerically** |
| 9 | Folding the species contraction into the radial weights: **1.61× faster** edge-feature construction, **49× fewer FLOPs**, 57× smaller intermediate — bit-exact | VMK 5.6(14) | **verified + benchmarked** |
| 10 | Natural-parity truncation *is* the Gaunt selection rule `l1+l2+L` even — a stronger justification than the manuscript gives | VMK 5.6(9),(14) | verified |
| 11 | `∇[f(r)Y_lm]` has a closed irreducible form → analytic force kernel for the AOT/LAMMPS path, and an independent check on autograd | VMK 5.8(9) | derived |
| 12 | Multipole expansion in the existing `l`-channels, if long-range electrostatics is ever added | VMK 5.17(21) | derived |

### Finding 8 in detail -- the largest single result so far

`ACEV2Descriptor._density` computes
`edge_features = tp_density(node_attrs[sender], harmonics, radial_net(radial))`.
Because `tp_density`'s first argument is **all scalars** (`64x0e`), VMK 5.6(14)
says the whole operation collapses to manuscript Eq. (12),

```
a^(l)_{e,c,m} = coef(e,c) · Y_lm(r̂_e) ,    coef(e,c) = Σ_q W2[h,q,c] hidden_h(d_e) E[s(e),q]
```

*Verified normalization-free:* for every edge and every `l > 0`, the `(c,m)`
matrix of `edge_features` has **rank 1** (max 2nd/1st singular value
3.4e-16 over 608 edges). The recovered per-block constant is exactly
`1/√(2l+1)`, matching the `C^{L0}_{l1 0 l2 0}` factor of VMK 5.6(14).

The model therefore depends on the final radial layer `W2` (32×1792 = 57,344
params) and the species embedding `E` **only through**

```
M[s,h,c] = Σ_q W2[h,q,c] E[s,q]        with only  n_species × 32 × 28  d.o.f.
```

| `n_species` | `W2` + used `E` | d.o.f. in `M` | redundant |
|---|---|---|---|
| 2 (water; C/H reaction) | 57,472 | 1,792 | **96.9%** |
| 3 (CsPbI3) | 57,536 | 2,688 | **95.3%** |
| 4 | 57,600 | 3,584 | 93.8% |
| 8 | 57,856 | 7,168 | 87.6% |

*Verified empirically:* a 50× perturbation of `W2` confined to the null space of
`W2 → M` changes the output by **9.2e-13** (output scale 1.96). For CsPbI3 that
is **54,656 parameters = 41% of the 132,005-parameter model** that cannot
affect any prediction.

Combined with Finding 1 (4,136 dead weights in `contractions[0]`),
**≈ 44% of TRACE-v2's advertised parameter count is structurally
unidentifiable** for the three systems reported in the manuscript.

*Why this is a speed **and** accuracy result.* Speed: the current path
materializes a `[L, 1792]` per-edge weight tensor (1.09M floats for 608 edges)
and does a 34.9 MFLOP matmul; the factorized path precomputes `M` once per
species and does a 0.72 MFLOP matmul on a `[L, 32]` intermediate — measured
**1.61× faster** on this operation, bit-exact (rel. diff 5.7e-16). Accuracy:
Muon and AdamW are currently spending 41% of their update budget, and all of
the weight decay applied to `W2`, on directions that provably do nothing;
re-parameterizing directly in terms of `M` gives the same function class with
2,688 parameters, better conditioning, and far less overfitting surface on
863 training structures.

### Honest negatives -- speed hypotheses that did *not* pan out

Recorded so they are not re-tried.

| Hypothesis | Measured | Verdict |
|---|---|---|
| Replacing `tp_density` with a hand-written per-`l` loop (keeping per-edge weights) | 2.281 → 2.237 ms (**1.02×**) | no gain; e3nn's TP is not the bottleneck by itself — the *per-edge weight tensor* is |
| Fusing the `H` per-head `o3.Linear` value projections into one | H=2: 1.13×; H=4: 1.01×; **H=8: 0.98×** | not worth it; actively worse at large `H` |
| Addition theorem (VMK 5.17(9)) to replace `Σ_m` contractions with Legendre polynomials | cost `n·(l+1)²` vs `n²·l`; wins only when `l_max ≳ n̄` | **not a win at `n̄ ≈ 15`, `l_max = 2`** — useful as an exact test identity, not an optimization |

*Profile of the real forward pass* (40-atom cell, 608 edges, 1 CPU thread,
energy+forces = 20.8 ms): `aten::bmm` 38.4% of self CPU (1440 calls),
`aten::einsum` 21.4% of total (600 calls), `aten::fill_` 6.4%, `aten::mm` 5.1%.
The batched-matmul time is dominated by the per-edge weight contraction that
Finding 8 removes.

---

## Reading plan

Chapters ordered by value to TRACE. Table sections are deliberately excluded --
`e3nn` computes those coefficients, and reading them verbatim is neither
achievable in context nor useful.

| Priority | Chapter / section | Printed pp. | PDF pp. | Why it matters |
|---|---|---|---|---|
| **done** | 3. Irreducible tensors (all) | 61-71 | 74-84 | ACE coupling, recoupling, redundancy → Findings 1-7 |
| **done** | 5.1-5.12, 5.16-5.17 Spherical harmonics | 130-153, 160-169 | 143-166, 173-182 | conventions, Gaunt, bipolar, addition theorem → Findings 8-12 |
| 1 | 11. Graphical method (all) | 412-451 | 425-464 | systematic derivation & simplification of contraction networks; could automate basis enumeration and find further redundancies like Findings 1 and 8 |
| 2 | 8.1-8.11 CG coefficients: definition, symmetry, recursion, sums of products | 235-269 | 248-282 | symmetry relations → further weight-redundancy audits |
| 3 | 12. Sums involving vector-addition and recoupling coefficients | 452-474 | 465-487 | identities to collapse/simplify TRACE contractions |
| 4 | 13. Matrix elements of irreducible tensor operators (Wigner-Eckart) | 475-504 | 488-517 | principled equivariant readout / tensor-operator heads |
| 5 | 7. Tensor spherical harmonics (esp. 7.3 vector SH) | 196-234 | 209-247 | needed to implement the analytic force kernel of Finding 11 |
| 6 | 9.1-9.10 6j & Racah; 10.1-10.10 9j | 290-309; 333-359 | 303-322; 346-372 | the recoupling coefficients used by Ch. 3.3 |
| 7 | 1.4 Rotations of coordinate system; 4.1-4.12 Wigner D-functions | 21-35; 72-97 | 34-48; 85-110 | Euler conventions for equivariance tests |
| — | skip | — | — | Ch. 2 (spin operators), Ch. 6 (spin functions), 5.13-5.15 (zeros/extrema), all numerical/algebraic tables |

Sections 1 and 2 are the ones most likely to *generate new architecture ideas*
rather than audit the existing one.
