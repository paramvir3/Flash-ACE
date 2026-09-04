# TRACE v2 tensor dimensions and local attention

This note explains the tensor shapes in the current TRACE v2 model, with the
CsPbI3 training configuration as a concrete example. In particular, it
clarifies why one may see a width of 240 in the attention block even though
the raw ACE edge token has only 60 components. It also explains the sparse
receiver-wise implementation, its exact relationship to standard
cross-attention, the sense in which the block is a transformer, and the prior
work that must be cited.

## 1. Configuration used in the example

The relevant settings in `training/config.yaml` are

```yaml
r_max: 6
l_max: 2
num_radial: 12
hidden_dim: 64
num_layers: 1
architecture_version: 2
correlation_order: 4
correlation_channels: 16
attention_num_heads: 2
attention_key_dim: null
```

For the inspected 40-atom CsPbI3 frame, these settings produced

```text
atomic numbers, z          [40]
edge_index                 [2, 622]
edge_vectors               [622, 3]
raw ACE edge features      [622, 60]
center representation      [40, 240]
queries                    [40, 2, 32]
keys                       [622, 2, 32]
attention logits/weights   [622, 2]
one-head values            [622, 240]
energy scalar channels     [40, 64]
```

The number 240 is therefore the width of the center representation and of
each projected attention value. It is **not** the width of the raw edge
descriptor.

## 2. What `edge_index` and `edge_vectors` represent

For \(E\) directed neighbor edges, `edge_index` has shape \([2,E]\). Each
column identifies one directed edge \(j\rightarrow i\):

```text
edge_index[0, e] = sender atom j
edge_index[1, e] = receiver/center atom i
```

The corresponding periodic displacement is

\[
\mathbf r_{ji}
=
\mathbf r_j-\mathbf r_i+\mathbf n_{ji}^{\mathsf T}\mathbf H ,
\]

where \(\mathbf H\) is the periodic cell and \(\mathbf n_{ji}\) is the
integer cell shift stored for that edge. Thus `edge_vectors[e]` contains
three Cartesian components and has shape \([E,3]\).

For the example,

\[
N=40,\qquad E=622,
\]

so

```text
edge_index   [2, 622]
edge_vectors [622, 3]
```

means that the neighbor list contains 622 directed periodic neighbor
relations within the \(6\)-Angstrom cutoff. It does not mean that there are
622 atoms. The average number of incoming directed edges in this frame is

\[
\frac{622}{40}=15.55.
\]

## 3. Meaning of a scalar, vector, and tensor channel

TRACE organizes features into irreducible representations, or irreps, of
\(O(3)\). An irrep is labeled by angular degree \(\ell\) and parity.

For a rotation or reflection \(R\), an order-\(\ell\) feature transforms as

\[
\mathbf x^{(\ell)} \longmapsto
D^{(\ell)}(R)\mathbf x^{(\ell)}.
\]

Here \(D^{(\ell)}(R)\) is the representation matrix for angular degree
\(\ell\). The number of components in one copy is

\[
\dim(\ell)=2\ell+1.
\]

The current \(l_{\max}=2\) representation contains

| e3nn label | Physical type | Components per copy | Transformation |
|---|---:|---:|---|
| `0e` | even scalar | 1 | unchanged by rotations and inversion |
| `1o` | polar vector | 3 | rotates as a vector and changes sign under inversion |
| `2e` | rank-2, traceless tensor irrep | 5 | transforms with \(D^{(2)}(R)\) and is even under inversion |

The word **scalar** has a representation-theoretic meaning here. A scalar
channel is invariant under every \(O(3)\) transformation:

\[
s' = s.
\]

By contrast, the three stored components of one `1o` copy must be treated
together:

\[
\mathbf v' = R\mathbf v.
\]

Although every tensor component is stored as a floating-point number, the
components of `1o` and `2e` are not independent invariant scalars.

## 4. Construction of the raw ACE edge token

### 4.1 Species embedding

The sender species \(Z_j\) is mapped to a learned invariant embedding

\[
\mathbf e_{Z_j}\in 64\times 0e.
\]

Thus every atom starts with 64 learned scalar channels. The order of species
in an external LAMMPS type map determines how LAMMPS types are converted to
atomic numbers, but it does not change the mathematical channel dimensions.

### 4.2 Radial and angular functions

For every directed edge \(j\rightarrow i\), TRACE evaluates 12 radial basis
functions,

\[
\mathbf R(r_{ji}) =
\left(R_1(r_{ji}),\ldots,R_{12}(r_{ji})\right),
\]

and real spherical harmonics through \(\ell=2\),

\[
\mathbf Y(\widehat{\mathbf r}_{ji})
=
\bigoplus_{\ell=0}^{2}
\mathbf Y^{(\ell)}(\widehat{\mathbf r}_{ji}).
\]

The radial basis is multiplied by the compact \(C^2\) cutoff

\[
f_{\mathrm{cut}}(r)=
\begin{cases}
(1-x)^3(1+3x+6x^2), & x=r/r_{\max}<1,\\
0, & x\geq 1.
\end{cases}
\]

The function and its first two radial derivatives vanish at \(r_{\max}\).
This smoothness is important for continuous energies, forces, and stresses.

The 12 radial values are not appended directly to produce a
\(12+\cdots\)-dimensional edge vector. A small radial network maps them to
the path weights of an equivariant tensor product.

### 4.3 Equivariant edge density contribution

The raw edge token is

\[
\mathbf a_{ji}
=
\operatorname{TP}
\left[
\mathbf e_{Z_j},
\mathbf Y(\widehat{\mathbf r}_{ji});
\mathbf W\!\left(\mathbf R(r_{ji})\right)
\right],
\]

where `TP` is an \(O(3)\)-equivariant tensor product and
\(\mathbf W\) denotes the radial network.

With `correlation_channels: 16` and \(l_{\max}=2\), its irreps are

\[
\mathcal V_{\mathrm{edge}}
=16\times 0e
\oplus 8\times 1o
\oplus 4\times 2e.
\]

Its flattened numerical width is therefore

\[
\begin{aligned}
D_{\mathrm{edge}}
&=16(2\cdot 0+1)
 +8(2\cdot 1+1)
 +4(2\cdot 2+1)\\
&=16+24+20\\
&=60.
\end{aligned}
\]

For 622 edges, the raw edge tensor consequently has shape

```text
[622, 60]
```

and its blocks are

```text
components  0:16   -> 16 copies of 0e
components 16:40   ->  8 copies of 1o, with 3 components per copy
components 40:60   ->  4 copies of 2e, with 5 components per copy
```

Only the first 16 components are invariant edge scalars.

## 5. Local ACE density and body-order correlations

The edge contributions are summed over the fixed neighbor environment of
center \(i\):

\[
\mathbf A_i
=
\sum_{j\in\mathcal N(i)}
\mathbf a_{ji}.
\]

Because this is a sum, \(\mathbf A_i\) is invariant to a permutation of the
neighbor-list ordering and equivariant under \(O(3)\).

TRACE then recursively couples this same local density with itself:

\[
\begin{aligned}
\mathbf C_i^{(2)} &= \mathbf A_i,\\
\mathbf C_i^{(3)} &=
\operatorname{TP}_2(\mathbf C_i^{(2)},\mathbf A_i),\\
\mathbf C_i^{(4)} &=
\operatorname{TP}_3(\mathbf C_i^{(3)},\mathbf A_i).
\end{aligned}
\]

The superscript denotes body order including the central atom. The
configuration `correlation_order: 4` therefore retains density, three-body,
and four-body correlations. Learned equivariant linear maps mix the retained
channels into the center representation:

\[
\mathbf h_i^{(0)}
=
\mathbf h_i^{\mathrm{center}}
+\sum_{\nu=2}^{4}L_\nu\mathbf C_i^{(\nu)}.
\]

No updated hidden state from atom \(j\) enters this construction. All
neighbor information is generated directly from species and relative
geometry inside the cutoff.

## 6. Why the center representation has width 240

The output multiplicities are controlled by `hidden_dim: 64`:

\[
\mathcal V_{\mathrm{node}}
=64\times 0e
\oplus 32\times 1o
\oplus 16\times 2e.
\]

Its flattened width is

\[
\begin{aligned}
D_{\mathrm{node}}
&=64(1)+32(3)+16(5)\\
&=64+96+80\\
&=240.
\end{aligned}
\]

The 240 stored components therefore mean

```text
64 invariant scalar channels
32 vector copies x 3 components = 96 components
16 l=2 copies x 5 components    = 80 components
total                            = 240 components
```

This is a direct sum of irreducible tensor channels, not a list of 240
unrelated scalar descriptors.

## 7. Strict local equivariant attention

The single v2 attention layer performs cross-attention from each center
representation to its fixed raw edge tokens.

### 7.1 Queries

Queries are formed only from the 64 invariant scalar channels of the center:

\[
\mathbf q_i^{(h)}
=W_Q^{(h)}\mathbf h_{i,0e},
\]

where \(h=1,2\) labels the two heads. Since `attention_key_dim` is `null`,
the code chooses

\[
d_k=\max(8,64/2)=32.
\]

For 40 atoms, the query shape is consequently

```text
[40, 2, 32]
```

### 7.2 Keys

Keys are formed from the 16 invariant `0e` components of each raw edge token:

\[
\mathbf k_{ji}^{(h)}
=W_K^{(h)}\mathbf a_{ji,0e}.
\]

For 622 edges, their shape is

```text
[622, 2, 32]
```

The keys do not contain an updated hidden state of the sender atom. This is
the precise sense in which v2 uses fixed-environment cross-attention rather
than graph self-attention or message passing between learned atomic states.

### 7.3 Invariant attention logits

For each edge and head, the score is

\[
\ell_{ji}^{(h)}
=
\frac{
\mathbf q_i^{(h)}\cdot\mathbf k_{ji}^{(h)}
}{\sqrt{d_k}}
+b_h\!\left(r_{ji}\right)
-\operatorname{softplus}(\lambda_h)r_{ji}.
\]

Every term is invariant under \(O(3)\):

- the query and key are built from invariant scalar channels;
- their dot product is invariant;
- the radial bias depends only on distance;
- the distance penalty depends only on distance.

Thus the attention weights cannot introduce a preferred orientation.

### 7.4 How the code computes the query-key dot product

The corresponding line in `StrictLocalEquivariantAttentionBlock.forward` is

```python
logits = (
    queries[receiver] * keys
).sum(dim=-1) / math.sqrt(self.key_dim)
```

Before radial terms are added, it evaluates

\[
s_e^{(h)}
=
\frac{
\mathbf q_{r(e)}^{(h)}
\cdot
\mathbf k_e^{(h)}
}{
\sqrt{d_k}
},
\]

where

- \(e=1,\ldots,L\) labels a directed edge;
- \(r(e)\) is the receiver, or center, of edge \(e\);
- \(h=1,\ldots,H\) labels an attention head;
- \(d_k\) is the dimension of one query or key.

Here \(L\) denotes the same directed-edge count denoted by \(E\) in the
neighbor-list discussion above.

For the CsPbI3 example,

```text
queries             [40, 2, 32]   = [N, H, d_k]
receiver            [622]         = [L]
queries[receiver]   [622, 2, 32]  = [L, H, d_k]
keys                [622, 2, 32]  = [L, H, d_k]
```

Indexing by `receiver` gathers the correct center query for every edge. If,
for example,

```text
receiver = [0, 0, 0, 1, 1, 2, ...],
```

then

```text
queries[receiver] =
[
    queries[0],
    queries[0],
    queries[0],
    queries[1],
    queries[1],
    queries[2],
    ...
].
```

The elementwise multiplication produces

```text
queries[receiver] * keys   [L, H, d_k],
```

and `sum(dim=-1)` contracts the key dimension:

\[
\sum_{a=1}^{d_k}
q_{r(e),ha}k_{e,ha}.
\]

The result has shape

```text
logits   [L, H].
```

This operation can equivalently be written as

```python
logits = torch.einsum(
    "lhd,lhd->lh",
    queries[receiver],
    keys,
) / math.sqrt(self.key_dim)
```

or as the explicit loop

```python
for edge in range(num_edges):
    center = receiver[edge]
    for head in range(num_heads):
        logits[edge, head] = torch.dot(
            queries[center, head],
            keys[edge, head],
        ) / math.sqrt(key_dim)
```

For two row vectors \(\mathbf q\) and \(\mathbf k\),

\[
\mathbf q\mathbf k^{\mathsf T}
=
\sum_{a=1}^{d_k}q_a k_a.
\]

Therefore, `(q * k).sum(dim=-1)` is precisely the same vector dot product as
\(qk^{\mathsf T}\). An explicit transpose is needed when constructing many
query-key combinations with a matrix multiplication. It is unnecessary when
the desired query and key vectors have already been aligned pair by pair.

The factor \(1/\sqrt{d_k}\) is the usual scaled-dot-product normalization. If
the query and key components have comparable variance, the variance of their
unscaled sum grows approximately as \(d_k\). Dividing by \(\sqrt{d_k}\)
keeps the logit scale approximately independent of key width and helps avoid
an excessively saturated softmax.

### 7.5 Why the logits are `[L, H]`, not `[L, L]`

The general scaled-dot-product attention equation is

\[
\mathbf S
=
\frac{\mathbf Q\mathbf K^{\mathsf T}}{\sqrt{d_k}},
\qquad
\mathbf Q\in\mathbb R^{n_q\times d_k},
\qquad
\mathbf K\in\mathbb R^{n_k\times d_k}.
\]

Its output shape is

\[
\mathbf S\in\mathbb R^{n_q\times n_k}.
\]

The matrix is square only when the query and key sequences have the same
length. In ordinary self-attention over \(L\) tokens,

\[
n_q=n_k=L,
\]

so the score matrix has shape \([L,L]\).

TRACE v2 does not use edge-to-edge self-attention. It uses one query for each
atomic center and one key/value token for each directed edge:

\[
n_q=N,\qquad n_k=L.
\]

A naive dense calculation would therefore produce \([N,L]\) scores per
head, not \([L,L]\). Even this \([N,L]\) matrix would be mostly invalid,
because center \(i\) is allowed to attend only to edges whose receiver is
\(i\).

Define the center-edge mask

\[
M_{ie}
=
\begin{cases}
1, & r(e)=i,\\
0, & r(e)\ne i.
\end{cases}
\]

The conceptual dense score tensor is

\[
S_{ieh}
=
\frac{
\mathbf q_i^{(h)}
\cdot
\mathbf k_e^{(h)}
}{
\sqrt{d_k}
},
\qquad
S\in\mathbb R^{N\times L\times H},
\]

but only entries satisfying \(M_{ie}=1\) are valid. Every directed edge has
exactly one receiver, so there are exactly \(L\) valid center-edge pairs per
head. The implementation stores only those entries:

\[
s_{eh}=S_{r(e),e,h}.
\]

This compact representation has shape \([L,H]\).

Equivalently, one could form the dense tensor with

```python
q = queries.permute(1, 0, 2)  # [H, N, d_k]
k = keys.permute(1, 0, 2)     # [H, L, d_k]
dense = q @ k.transpose(-2, -1)  # [H, N, L]
```

and then mask every entry for which `receiver[edge] != center`. For every
valid edge,

```text
dense[head, receiver[edge], edge] == logits[edge, head]
```

before radial terms are added. Constructing and masking the dense tensor
would perform many unnecessary dot products.

#### A small ragged-neighborhood example

Suppose three centers have respectively two, three, and one incoming edges:

\[
n_0=2,\qquad n_1=3,\qquad n_2=1,\qquad L=6,
\]

with

```text
receiver = [0, 0, 1, 1, 1, 2].
```

For one attention head, the conceptual masked center-edge matrix is

\[
\mathbf S=
\begin{pmatrix}
s_0&s_1&-\infty&-\infty&-\infty&-\infty\\
-\infty&-\infty&s_2&s_3&s_4&-\infty\\
-\infty&-\infty&-\infty&-\infty&-\infty&s_5
\end{pmatrix}.
\]

The sparse representation stores only

\[
[s_0,s_1,s_2,s_3,s_4,s_5].
\]

The segment softmax later normalizes the three receiver groups separately:

\[
(s_0,s_1),\qquad
(s_2,s_3,s_4),\qquad
(s_5).
\]

The edges need not be stored contiguously or sorted. The `receiver` index
identifies the correct group.

#### What an `[L, L]` matrix would mean

An \([L,L]\) matrix would correspond to every edge acting as both a query
and a key:

\[
S_{ee'}^{(h)}
=
\frac{
\mathbf q_e^{(h)}
\cdot
\mathbf k_{e'}^{(h)}
}{
\sqrt{d_k}
}.
\]

That is global edge self-attention. It would allow an edge in one atomic
environment to attend to every edge in the entire system, would break the
desired strict locality unless heavily masked, and would require
\(O(L^2Hd_k)\) work and \(O(L^2H)\) logit memory.

A different but still local architecture could let the \(n_i\) edges around
each center attend to one another. It would construct a separate
\([n_i,n_i]\) matrix for each center and require work proportional to

\[
Hd_k\sum_i n_i^2.
\]

That would be local edge-to-edge self-attention, but it is not the current
TRACE v2 block.

TRACE instead uses one environment-conditioned center query to select among
the fixed edge tokens in that center's neighborhood. Since the center
representation already contains the local ACE density correlations, its
query depends on the complete local environment even though no edge-edge
score matrix is constructed. The logit calculation has cost

\[
O(LHd_k)
\]

and stores \(O(LH)\) scores.

The block is therefore accurately described as **sparse local
center-to-edge cross-attention**. It is not global edge self-attention and it
does not exchange updated sender-atom hidden states.

### 7.6 Cutoff-weighted normalization

For incoming edges of center \(i\), TRACE evaluates

\[
\alpha_{ji}^{(h)}
=
\frac{
f_{\mathrm{cut}}(r_{ji})
\exp\!\left(\ell_{ji}^{(h)}\right)
}{
1+\displaystyle\sum_{k\in\mathcal N(i)}
f_{\mathrm{cut}}(r_{ki})
\exp\!\left(\ell_{ki}^{(h)}\right)
}.
\]

The unit term is a null/self channel. It prevents softmax normalization from
cancelling the cutoff when all neighbors approach \(r_{\max}\). The
implementation uses a shifted, numerically stable form of this equation.

The attention weights have shape

```text
[622, 2]
```

so each directed edge receives one invariant coefficient per head.

### 7.7 Equivariant values

Each head maps the complete 60-component edge token to the
240-component node irreps:

\[
\mathbf v_{ji}^{(h)}
=L_V^{(h)}\mathbf a_{ji},
\qquad
L_V^{(h)}:
\mathcal V_{\mathrm{edge}}\rightarrow\mathcal V_{\mathrm{node}}.
\]

The map \(L_V^{(h)}\) is an e3nn equivariant linear map. Consequently, one
head produces

```text
[622, 240]
```

values. The invariant attention weights multiply whole irrep copies without
changing their transformation law. The local update is

\[
\Delta\mathbf h_i
=
L_O\left[
\frac{1}{H}
\sum_{h=1}^{H}
\sum_{j\in\mathcal N(i)}
\alpha_{ji}^{(h)}
\mathbf v_{ji}^{(h)}
\right],
\qquad H=2.
\]

The residual update is

\[
\mathbf h_i'
=
\mathbf h_i^{(0)}
+\Lambda_{\mathrm{attn}}\Delta\mathbf h_i,
\]

where the learned layer scale is applied consistently within each irrep.

## 8. Scalar feed-forward update

After attention, TRACE computes invariant information from

1. the 64 existing `0e` channels, and
2. squared norms of every non-scalar irrep copy.

For example, a vector norm

\[
\lVert\mathbf v\rVert^2=v_x^2+v_y^2+v_z^2
\]

is invariant under \(O(3)\). These invariants enter a scalar feed-forward
network. Its output updates only the 64 scalar channels:

\[
\mathbf h_{i,0e}''
=
\mathbf h_{i,0e}'
+\Lambda_{\mathrm{FFN}}\,
\operatorname{FFN}
\left(
\mathbf h_{i,0e}',
\left\{\lVert\mathbf h_i^{(\ell,c)}\rVert^2\right\}_{\ell>0,c}
\right).
\]

The vector and rank-2 channels are not passed through an ordinary
component-wise MLP, because doing so would generally violate equivariance.

## 9. Invariant energy and conservative derivatives

Only the final 64 invariant scalar channels enter the atomic-energy readout:

\[
\varepsilon_i
=
\operatorname{MLP}_{E}(\mathbf h_{i,0e}''),
\qquad
E=\sum_i\varepsilon_i.
\]

The sum gives an extensive total energy. Forces are obtained from this same
scalar energy,

\[
\mathbf F_i=-\frac{\partial E}{\partial\mathbf r_i},
\]

and stress is obtained from the strain derivative of the same energy. Thus
the model does not use an independent force prediction in production, and
its forces are conservative up to numerical precision.

## 10. Complete dimensional flow for the example

```text
40 atomic numbers
    |
    v
species embedding
    [40, 64] = 64x0e
    |
    +-- 622 periodic directed edges within 6 Angstrom
    |       edge_index   [2, 622]
    |       edge_vectors [622, 3]
    |
    v
12 radial functions + spherical harmonics through l=2
    |
    v
raw equivariant edge tokens
    [622, 60] = 16x0e + 8x1o + 4x2e
    |
    +-----------------------------+
    |                             |
    v                             v
sum by center and              scalar edge block
ACE density correlations      [622, 16]
    |                             |
    v                             v
center tensors                 keys [622, 2, 32]
[40, 240]                         ^
64x0e + 32x1o + 16x2e             |
    |                              |
    +--> scalar center block ------+
         [40, 64]
             |
             v
         queries [40, 2, 32]

complete edge tokens --equivariant linear maps--> values [622, 240] per head
queries + keys + distances ---------------------> weights [622, 2]
weighted sum by receiver -----------------------> update [40, 240]
residual + invariant scalar FFN ----------------> final h [40, 240]
final 64 scalar channels -----------------------> atomic energies -> total E
```

## 11. General dimension formulas

For the current multiplicity rule and \(l_{\max}=2\), let

- \(C\) be `correlation_channels`;
- \(H\) be `hidden_dim`.

Then the edge multiplicities are

\[
(m_0,m_1,m_2)=(C,C/2,C/4),
\]

and the raw edge width is

\[
D_{\mathrm{edge}}
=C+3(C/2)+5(C/4).
\]

The node multiplicities are

\[
(n_0,n_1,n_2)=(H,H/2,H/4),
\]

and the node width is

\[
D_{\mathrm{node}}
=H+3(H/2)+5(H/4).
\]

For \(C=16\) and \(H=64\), these give

\[
D_{\mathrm{edge}}=60,
\qquad
D_{\mathrm{node}}=240.
\]

Changing `num_radial` changes the radial resolution and the radial network,
but it does not directly set either flattened width. Changing the number of
attention heads changes the number of independent attention distributions,
but each head still maps values into the full node irrep space.

## 12. Relevant implementation locations

- Edge and node irreps:
  [`flashace/physics.py`](../flashace/physics.py), `ACEV2Descriptor.__init__`
- Radial basis and smooth cutoff:
  [`flashace/physics.py`](../flashace/physics.py), `SmoothACERadialBasis`
- Raw edge tokens and density sum:
  [`flashace/physics.py`](../flashace/physics.py), `ACEV2Descriptor._density`
- Recursive ACE correlations:
  [`flashace/physics.py`](../flashace/physics.py), `ACEV2Descriptor.forward`
- Queries, keys, values, and cutoff softmax:
  [`flashace/model.py`](../flashace/model.py),
  `StrictLocalEquivariantAttentionBlock`
- Energy, force, and stress path:
  [`flashace/model.py`](../flashace/model.py), `TransformersACE.forward`

The separate file `flashace/attention.py` is not used by the current TRACE v2
forward path. The active attention implementation is
`StrictLocalEquivariantAttentionBlock` in `flashace/model.py`.

## 13. Why accumulation by `receiver` is correct

The attention update is accumulated with

```python
update.index_add_(
    0,
    receiver,
    alpha[:, head : head + 1] * values,
)
```

This is not an indexing error. Each directed token \(e=(j\rightarrow i)\)
describes atom \(j\) as observed from center \(i\). Its value must therefore
contribute to the representation of \(i\):

\[
\Delta\mathbf h_i
=
\sum_{e:r(e)=i}
\alpha_e\mathbf v_e
=
\sum_{j\in\mathcal N(i)}
\alpha_{ji}\mathbf v_{ji}.
\]

The `receiver` array supplies the map \(e\mapsto i\), and `index_add_` performs
this ragged neighborhood sum. A full directed neighbor list also contains the
reverse relation \(i\rightarrow j\), with the appropriate periodic
displacement, when the two atoms are mutual neighbors. That reverse token is
accumulated into center \(j\). The two directed relations are distinct local
observations and should not be combined into one undirected update.

The sender index is used earlier when the edge geometry, sender species, and
edge-density token are constructed. It is intentionally absent from the
attention accumulation because TRACE does not update the current hidden state
of atom \(j\) through the token \(j\rightarrow i\), nor does it place
\(\mathbf h_j^{(t)}\) in the key or value. For one layer,

\[
\mathbf h_i^{(0)}
\longrightarrow
\mathbf q_i,\qquad
\mathbf a_{ji}
\longrightarrow
(\mathbf k_{ji},\mathbf v_{ji}),\qquad
(\mathbf q_i,\mathbf k_{ji},\mathbf v_{ji})
\longrightarrow
\Delta\mathbf h_i .
\]

The residual update then gives

\[
\mathbf h_i^{(1)}
=
\mathbf h_i^{(0)}+\Delta\mathbf h_i.
\]

Thus `queries[receiver]` selects the query whose environment is being updated,
while `index_add_(..., receiver, ...)` returns the weighted values to that same
environment. These are the gather and scatter operations for one
center-to-neighbor-set cross-attention calculation.

## 14. Relation to standard, sparse, and windowed attention

### 14.1 The local operation is ordinary cross-attention on a ragged set

For center \(i\), collect its \(n_i\) incoming edge keys and values into

\[
\mathbf K_i\in\mathbb R^{n_i\times d_k},
\qquad
\mathbf V_i\in\mathbb R^{n_i\times D_v},
\]

and write its one query as

\[
\mathbf Q_i\in\mathbb R^{1\times d_k}.
\]

Standard cross-attention for this center is

\[
\mathbf s_i
=
\frac{\mathbf Q_i\mathbf K_i^{\mathsf T}}{\sqrt{d_k}}
\in\mathbb R^{1\times n_i},
\qquad
\Delta\mathbf h_i
=
\operatorname{softmax}(\mathbf s_i)\mathbf V_i .
\]

TRACE evaluates these equations for all centers simultaneously. Because the
neighborhood sizes \(n_i\) differ, the code stores the local score rows
back-to-back in one edge array rather than padding every environment to the
same width. Since

\[
L=\sum_{i=1}^{N}n_i,
\]

the compact score tensor has shape \([L,H]\). A segment softmax applies the
normalization independently to all entries sharing the same receiver. The
operation is mathematically equivalent to a collection of \(N\) local
cross-attention calculations, apart from TRACE's explicit radial terms,
cutoff factor, and null channel.

### 14.2 What is sparse

There are two distinct sparsity statements:

1. **Physical neighbor sparsity.** Of the possible ordered atomic relations,
   only periodic images within \(r_{\max}\) become directed edge tokens.
2. **Attention-incidence sparsity.** Query \(i\) is paired only with edge
   tokens whose receiver is \(i\).

For \(N=40\), the \(N^2=1600\) count is the number of ordered atom-index pairs
before removing self pairs, applying the cutoff, and accounting for periodic
images. The observed \(L=622\) is the number of directed periodic neighbor
entries produced by the actual neighbor list. It is not the dimension of a
hidden dense atom-attention matrix.

After those \(622\) edge tokens have been constructed, TRACE also does not
perform all \(622^2\) edge-to-edge attention comparisons. Instead, each edge
key is paired with exactly one center query, namely the query of its receiver.
The number of stored attention logits is consequently \(LH\).

It is reasonable to call the operation **cutoff-windowed** because every
center attends only inside a real-space cutoff window. The more precise name
is **sparse local center-to-edge cross-attention**. It is not standard
atom-token windowed self-attention, because atoms are not simultaneously used
as the query, key, and value tokens. It is also not local edge-to-edge
self-attention, which would require \(\sum_i n_i^2\) scores per head.

### 14.3 Why no `[L,L]` tensor is required

The expression

\[
\mathbf Q\mathbf K^{\mathsf T}
\]

does not prescribe a square matrix. Its shape is \(n_q\times n_k\), and a
square result occurs only in self-attention when \(n_q=n_k\). Within one TRACE
environment, \(n_q=1\) and \(n_k=n_i\). The line

```python
(queries[receiver] * keys).sum(dim=-1)
```

computes precisely the valid entries of those \(N\) local
\(1\times n_i\) products. Forming a global \([L,L]\) matrix would define a
different model in which every directed edge queries every other directed
edge.

## 15. Is the TRACE v2 block a transformer?

The block contains the principal components of a transformer-style
cross-attention block:

- learned query, key, and value projections;
- scaled dot-product logits;
- multiple attention heads;
- a normalized weighted value sum;
- pre-normalization and a residual attention update;
- a residual feed-forward update.

It is therefore technically defensible to describe TRACE as using a
transformer block. The precise description is

> a strictly local, equivariant, center-to-edge cross-attention transformer
> block over fixed ACE density tokens.

The qualifier matters. TRACE v2 is not a conventional sequence transformer,
global self-attention model, or Equiformer-style graph transformer. In the
latter class, messages, keys, or values generally depend on the current
learned states of both endpoint atoms. In TRACE v2,

\[
\mathbf q_i^{(t)}=Q(\mathbf h_i^{(t)}),\qquad
\mathbf k_{ji}=K(\mathbf a_{ji}),\qquad
\mathbf v_{ji}=V(\mathbf a_{ji}),
\]

where the directed edge token \(\mathbf a_{ji}\) is fixed after the local ACE
density construction. In particular,

\[
\frac{\partial\mathbf k_{ji}}
     {\partial\mathbf h_j^{(t)}}=0,
\qquad
\frac{\partial\mathbf v_{ji}}
     {\partial\mathbf h_j^{(t)}}=0.
\]

No attention-updated sender state is transmitted between atomic centers. If
several TRACE attention blocks are stacked, later queries can refine the
nonlinear response of center \(i\) to the same fixed-radius token set, but the
receptive field does not grow with layer count. For the concrete configuration
at the start of this document, `num_layers: 1`, so that trained model uses one
such block. Statements about increasing attention depth describe the supported
architecture, not that particular checkpoint.

The safest terminology in a manuscript is therefore:

- **TRACE architecture:** an ACE-conditioned local equivariant transformer;
- **attention mechanism:** fixed-environment center-to-edge cross-attention;
- **information flow:** local set aggregation without updated sender-state
  propagation.

Calling the operation simply "standard self-attention" or "graph
self-attention" would be inaccurate.

## 16. Origin of the attention construction and required citations

### 16.1 Provenance

No single publication contains the complete TRACE v2 block verbatim. The
implementation was assembled in this project to satisfy four simultaneous
requirements:

1. the center query must contain local ACE density correlations;
2. keys and values must remain functions only of a fixed directed edge token;
3. scalar invariant coefficients must weight equivariant tensor values;
4. computation must remain linear in the number of retained neighbor edges.

The first repository commits containing
`StrictLocalEquivariantAttentionBlock` are the June 21, 2026 TRACE v2
architecture commits. This repository history records implementation
provenance, but it is not by itself proof of scientific priority. The
mathematical ingredients have clear prior art and should be cited explicitly.

### 16.2 Citation map

| Implemented ingredient | Prior work that should be cited | Relationship to TRACE |
| --- | --- | --- |
| Scaled dot-product attention, multihead projections, residual attention, and feed-forward sublayer | Vaswani *et al.*, *Attention Is All You Need* | TRACE uses the standard scaled query-key dot product and transformer-block organization. |
| A query aggregating an unordered set of input tokens | Lee *et al.*, *Set Transformer* | The pooling-by-multihead-attention construction is the closest general set-attention precedent. TRACE replaces a learned seed query with an ACE-conditioned atomic-center query and uses a different local token set. |
| Masking or sparse normalization over a graph neighborhood | Veličković *et al.*, *Graph Attention Networks* | GAT is relevant to local neighborhood attention and sparse storage, although its attention rule and node-state message passing differ from TRACE. |
| Invariant attention weights acting on equivariant geometric values | Fuchs *et al.*, *SE(3)-Transformer* | Establishes geometric attention in which invariant coefficients combine equivariant features. Its graph messages depend on learned endpoint representations. |
| Transformer operations on irreducible \(O(3)\) tensor features | Liao and Smidt, *Equiformer* | Close equivariant-transformer precedent, but Equiformer constructs graph messages from current endpoint states. |
| Atomistic neighborhood attention combining invariant features and equivariant spherical-harmonic variables | Frank, Unke, and Müller, *SO3krates* | The closest atomistic attention precedent. SO3krates propagates current atomic features and updates its Euclidean variables, whereas TRACE fixes its edge-token key and value path. |
| Spherical-harmonic neighbor density and higher-body correlations | Drautz, *Atomic cluster expansion for accurate and transferable interatomic potentials* | Supplies the ACE representation-theoretic foundation; it is not an attention contribution. |
| Strict locality without atom-centered message passing | Musaelian *et al.*, *Allegro* | Establishes the strict-locality and scalable-decomposition precedent. Allegro evolves ordered-pair latent features rather than performing TRACE's fixed-token center cross-attention. |
| Smooth radius-modulated equivariant attention | Liao *et al.*, *EquiformerV3* | Prevents a general claim that smooth cutoff attention is new. TRACE's particular null-normalized cutoff envelope is an implementation-specific construction. |

The public `thorben-frank/mlff` repository also contains an experimental
`So3krataceLayer` combining SO3krates-style attention with
symmetry-adapted higher-body contractions. It is therefore not defensible to
claim that TRACE is the first model to combine an ACE-type construction with
attention. The narrower distinction is TRACE's dependency constraint:
ACE-correlated center states provide the queries, while every key and value is
an immutable directed neighbor-density token and contains no
attention-updated sender state.

### 16.3 Recommended attribution in the manuscript

A concise and technically accurate Methods statement is:

> We employ scaled dot-product attention in a permutation-invariant local
> set-aggregation form related to attention pooling in Set Transformer.
> Following geometric attention architectures, invariant scalar coefficients
> weight equivariant tensor values. TRACE differs in its dependency structure:
> an ACE-correlated representation of the central environment supplies the
> query, whereas the keys and values are immutable directed ACE density tokens
> determined only by species and local geometry. Consequently, attention depth
> refines the response to a fixed local environment without transmitting
> attention-updated states between atomic centers.

This statement attributes the established attention and equivariance
ingredients while identifying the particular TRACE construction. TRACE
should not claim invention of scaled dot-product attention, equivariant
attention, ACE, strict locality, or attention-based local aggregation.

### 16.4 Primary references

1. A. Vaswani *et al.*,
   [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
   *Advances in Neural Information Processing Systems* **30** (2017).
2. J. Lee *et al.*,
   [Set Transformer: A Framework for Attention-based
   Permutation-Invariant Neural Networks](https://proceedings.mlr.press/v97/lee19d.html),
   *Proceedings of Machine Learning Research* **97**, 3744 (2019).
3. P. Veličković *et al.*,
   [Graph Attention Networks](https://arxiv.org/abs/1710.10903),
   *International Conference on Learning Representations* (2018).
4. F. B. Fuchs *et al.*,
   [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention
   Networks](https://arxiv.org/abs/2006.10503),
   *Advances in Neural Information Processing Systems* **33** (2020).
5. Y.-L. Liao and T. Smidt,
   [Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic
   Graphs](https://arxiv.org/abs/2206.11990),
   *International Conference on Learning Representations* (2023).
6. J. T. Frank, O. T. Unke, and K.-R. Müller,
   [SO3krates: Equivariant attention for interactions on arbitrary
   length-scales in molecular systems](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bcf4ca90a8d405201d29dd47d75ac896-Abstract-Conference.html),
   *Advances in Neural Information Processing Systems* **35** (2022).
7. J. T. Frank *et al.*,
   [A Euclidean transformer for fast and stable machine learned force
   fields](https://www.nature.com/articles/s41467-024-50620-6),
   *Nature Communications* **15** (2024).
8. R. Drautz,
   [Atomic cluster expansion for accurate and transferable interatomic
   potentials](https://doi.org/10.1103/PhysRevB.99.014104),
   *Physical Review B* **99**, 014104 (2019).
9. A. Musaelian *et al.*,
   [Learning local equivariant representations for large-scale atomistic
   dynamics](https://doi.org/10.1038/s41467-023-36329-y),
   *Nature Communications* **14**, 579 (2023).
10. Y.-L. Liao *et al.*,
    [EquiformerV3: Scaling Equivariant Transformers to Large-Scale
    Atomistic Systems](https://arxiv.org/abs/2604.09130), preprint (2026).
11. T. Frank *et al.*,
    [`mlff`: Machine Learning Force Fields](https://github.com/thorben-frank/mlff),
    public software repository.
