# 2D Shape Transformer – Matrix Operations in Computer Graphics

**Course:** Mathematical Foundations / Linear Algebra  
**Problem statement:** Basic Operations with Matrices

## Objective
Understand matrix operations by transforming 2D polygons and printing the calculations. Each column of a **2×N NumPy matrix** is one corner `(x, y)`. The project uses a menu-driven CLI and matplotlib only.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

Choose **0** for the automatic viva demo, then **q** to quit. Alternatively:

```bash
python main.py --demo
```

For a computer without a display:

```bash
MPLBACKEND=Agg python main.py --demo
```

Normal menu operations open plot windows and return to the menu straight away; the windows stay open until you close them or quit. Choosing a custom shape shows a live plot that updates as each point is typed. The demo never asks questions or opens windows: it saves 13 PNGs in `outputs/`. `outputs/demo_output.txt` contains a captured run.

## Matrix concepts and functions

| Matrix concept | Feature | Function |
|---|---|---|
| Addition | Translate every corner by the same offset | `matrix_ops.add`, `transformations.translate`, `translation_offset` |
| Subtraction | Undo translation | `matrix_ops.subtract` |
| Scalar multiplication | Uniform scaling | `matrix_ops.scalar_multiply`, `transformations.scale` |
| Matrix multiplication | Rotate, scale, reflect, shear | `matrix_ops.multiply`, `transformations.apply` |
| Diagonal matrices | Independent x/y scaling | `transformations.scale_xy` |
| Transpose and orthogonality | Rotation transpose equals inverse | `matrix_ops.transpose`, `rotation_orthogonal` |
| Determinant / cofactor expansion | Area change, mirroring, flattening | `matrix_ops.determinant`, `analysis.det_meaning` |
| Rank | Independent dimensions | `matrix_ops.rank` |
| Gauss–Jordan elimination | Inverse and undo | `matrix_ops.inverse_gauss_jordan`, `transformations.undo` |
| Gauss–Jordan, step by step | Menu option 10: plot every row operation and the final inverse | `matrix_ops.gauss_jordan_steps`, `visualize.gauss_jordan` |
| Noncommutativity | Compare AB and BA | `visualize.chain` |
| Linear equations | Recover a 2×2 transformation from point pairs | `analysis.find_transformation` |
| LU factorization | Solve the same system a second way | `analysis.find_transformation` |
| Four fundamental subspaces | Explain projection and lost directions | `analysis.subspaces` |
| Eigenvalues/eigenvectors | Directions preserved as lines | `analysis.eigen` |
| Singular value decomposition | Orthogonal → stretch → orthogonal stages | `analysis.svd` |
| Shoelace formula | Verify transformed polygon area | `analysis.area` |

## Files

- `matrix_ops.py`: manual loops, cofactor determinant, rank and RREF inverse.
- `transformations.py`: transformation constructors, application and undo.
- `analysis.py`: mathematical explanations and bonus features.
- `visualize.py`: plotting and PNG saving.
- `main.py`: input validation, menu, built-in shapes and full demo.
- `requirements.txt`: the four external dependencies.

## Sample output

```text
2x2 determinant: ad - bc = 0.0
Determinant: 0.0
Verified: True
Area scale factor = |det(T)| = 0.0
Flattened: area becomes zero.
Matrix is singular – not invertible
Information was lost: crushed coordinates cannot be recovered uniquely.
```

Checks use `np.allclose()` and print `Verified: True/False`. `AB equals BA: False` is an expected explanation of noncommutativity, not a failed verification. Singular examples intentionally report no inverse, no solution, or infinitely many solutions.

## Mathematical notes for the viva

- With column vectors, `A @ B @ shape` applies **B first**, then A. Some pairs commute; the demo deliberately chooses rotation and unequal-axis scaling, which do not.
- Translation is affine. `translate(tx, ty)` returns a 2×1 offset column, repeated to 2×N before addition. Translation has no 2×2 transformation matrix; its linear part is identity. Undo subtracts the offset.
- Uniform scaling constructs `kI` by scalar multiplication; applying it is equivalent to multiplying every shape coordinate by k.
- `|det(T)|` is the area factor; a negative determinant reverses orientation. The shoelace formula assumes vertices follow a simple polygon's boundary order. Crossing edges give algebraic area and are outside this project's polygon assumption.
- Gauss–Jordan prints `[A | I]` and its RREF. SymPy rationalizes decimal strings to avoid spurious floating-point pivots. The left half must equal identity before returning the inverse.
- Point-pair recovery assumes a **linear** map through the origin. Two independent original vectors give four independent equations. Dependent vectors can give no solution or infinitely many solutions. Rank tests and numerical checks use floating-point tolerances.
- Eigenvectors stay on the same line: negative eigenvalues reverse them and zero eigenvalues crush them. A general rotation has complex eigenpairs, so the program explains why real arrows cannot be drawn.
- SVD's U and VT can reflect as well as rotate. Sigma contains nonnegative stretches. Projection loses information and cannot be uniquely undone.
- Menu option **10 (Inverse)** reads any square matrix and row-reduces `[A | I]` one elementary row operation at a time (swap, scale, eliminate) with exact fractions. Each step gets its own panel with the changed rows highlighted; the last panel shows `A⁻¹`. For a 2×2 matrix, a plot under each step shows the current left block applied to the current shape: it starts as `A × shape` and ends back at the original shape once the left block is `I`. The final plot shows `A⁻¹` applied to the shape. A missing pivot stops the reduction and reports that `A` is singular. The finished matrix is checked against SymPy's `rref()` and `np.linalg.inv`. The demo saves this as `outputs/inverse.png`.
- RREF is printed as an augmented matrix and final reduced matrix; it uses the class-notebook `Matrix().rref()` method rather than a custom row-reduction algorithm.

## Applications in CS

- **Computer graphics:** move, rotate, resize and reflect polygon vertices.
- **Game engines:** combine transformations to position objects and cameras.
- **Robotics:** coordinate transformations describe sensor and robot frames; translation needs affine/homogeneous treatment.
- **Image processing:** geometric warps resize, rotate and shear images.
- **AI / data science:** eigenvectors support principal component analysis; SVD supports dimensionality reduction and low-rank approximation.

This is a small educational project: no GUI framework, website or database.
