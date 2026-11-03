"""Simple matplotlib drawings; no GUI framework is needed."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent / 'outputs'


def draw(ax, original, transformed, title, T=None):
    for points, color, style, label in [(original, 'blue', '--', 'Original'),
                                        (transformed, 'red', '-', 'Transformed')]:
        closed = np.column_stack((points, points[:,0]))
        ax.plot(closed[0], closed[1], style, color=color, marker='o', label=label)
    ax.fill(transformed[0], transformed[1], color='red', alpha=0.2)
    if T is not None:
        title += '\nT = ' + np.array2string(T, precision=2) + f'\ndet(T) = {np.linalg.det(T):.3g}'
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(fontsize=8)


def finish(fig, filename=None):
    fig.tight_layout()
    if filename:
        OUTPUTS.mkdir(exist_ok=True)
        path = OUTPUTS / filename
        fig.savefig(path, dpi=140)
        print('Saved plot:', path.name)
        plt.close(fig)
    else:
        # Non-blocking: the menu continues while the window stays open.
        # Tk keeps the window responsive while the terminal waits for input.
        plt.show(block=False)
        plt.pause(0.1)


def draw_points(ax, shape, title, complete=True):
    """Draw a 2xN shape. While incomplete, the closing edge is dashed."""
    shape = np.asarray(shape, dtype=float).reshape(2, -1)
    n = shape.shape[1]
    if n:
        ax.plot(shape[0], shape[1], '-', color='blue', marker='o', label=title)
    if n >= 3:
        closing = shape[:, [-1, 0]]
        ax.plot(closing[0], closing[1], '-' if complete else ':', color='blue')
        ax.fill(shape[0], shape[1], color='blue', alpha=0.2 if complete else 0.08)
    for i, (x, y) in enumerate(shape.T, 1):
        ax.annotate(f'P{i} ({x:g}, {y:g})', (x, y), textcoords='offset points',
                    xytext=(6,6), fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.margins(0.15)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if n: ax.legend(fontsize=8)


def show_shape(shape, title='Shape', filename=None):
    fig, ax = plt.subplots(figsize=(7,6))
    draw_points(ax, shape, title)
    finish(fig, filename)


class LiveShape:
    """A non-blocking window that redraws while points are typed in the terminal."""

    def __init__(self, title='Custom shape (typing points...)'):
        self.title = title
        self.was_interactive = plt.isinteractive()
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7,6))
        self.update([])

    def update(self, points):
        self.ax.clear()
        draw_points(self.ax, np.array(points).T if points else np.empty((2,0)),
                    self.title, complete=False)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def close(self):
        plt.close(self.fig)
        if not self.was_interactive:
            plt.ioff()


def plot_shape(original, transformed, title, T=None, filename=None):
    fig, ax = plt.subplots(figsize=(7,6))
    draw(ax, original, transformed, title, T)
    finish(fig, filename)


def chain(shape, A, B, filename=None):
    # AB applies B first; BA applies A first (column-vector convention).
    AB, BA = A @ B, B @ A
    print('AB:\n', AB, '\nBA:\n', BA)
    print('AB equals BA:', np.allclose(AB, BA))
    print('Order can matter; some special pairs do commute.')
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    draw(axes[0], shape, AB @ shape, 'B first, then A: AB', AB)
    draw(axes[1], shape, BA @ shape, 'A first, then B: BA', BA)
    finish(fig, filename)
    return AB @ shape, BA @ shape


def draw_matrix(ax, M, title, split=None, changed=()):
    """Draw a (sympy) matrix as a grid of exact entries.

    split: column index of the [left | right] divider; changed: rows to highlight.
    """
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            y = rows - 1 - i
            color = ('#fff2a8' if i in changed else
                     '#e3edfb' if split is None or j < split else '#fbe3e3')
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=color, edgecolor='white'))
            ax.text(j + .5, y + .5, str(M[i, j]), ha='center', va='center', fontsize=11)
    if split is not None:
        ax.plot([split, split], [0, rows], color='black', linewidth=1.5)
        ax.text(split / 2, -.25, 'left block → I', ha='center', va='top', fontsize=8)
        ax.text((split + cols) / 2, -.25, 'right block → A⁻¹', ha='center', va='top', fontsize=8)
    ax.set_xlim(0, cols); ax.set_ylim(-.7, rows)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10)


def gauss_jordan(steps, inverse, shape=None, filename=None):
    """One panel per Gauss-Jordan step, plus a final panel showing A⁻¹.

    For 2x2 matrices a shape panel under each step shows (left block) @ shape:
    it starts as A @ shape and returns to the original shape once the left block is I.
    """
    n = steps[0][1].shape[0]
    geometric = shape is not None and n == 2
    panels = len(steps) + 1
    cols = min(panels, 4)
    grid = -(-panels // cols)
    per = 2 if geometric else 1
    matrix_height = .5 * n + 1.2
    fig, axes = plt.subplots(grid * per, cols, squeeze=False,
                             figsize=(4.2 * cols, grid * (matrix_height + 4 * geometric)),
                             gridspec_kw={'height_ratios': [matrix_height, 4] * grid}
                             if geometric else None)
    for ax in axes.flat:
        ax.axis('off')
    previous = None
    for k, (text, M) in enumerate(steps):
        r, c = divmod(k, cols)
        changed = () if previous is None else [i for i in range(n) if M.row(i) != previous.row(i)]
        draw_matrix(axes[r * per, c], M, f'Step {k}: {text}', split=n, changed=changed)
        if geometric:
            axes[r * per + 1, c].axis('on')
            L = np.array(M[:, :n], dtype=float)
            draw(axes[r * per + 1, c], shape, L @ shape, 'Left block × shape', L)
        previous = M
    r, c = divmod(len(steps), cols)
    ax = axes[r * per, c]
    if inverse is None:
        ax.text(.5, .5, 'No pivot available:\nA is singular,\nso A⁻¹ does not exist',
                ha='center', va='center', fontsize=12, color='darkred', transform=ax.transAxes)
    else:
        draw_matrix(ax, steps[-1][1][:, n:], 'Final result: A⁻¹')
        if geometric:
            axes[r * per + 1, c].axis('on')
            draw(axes[r * per + 1, c], shape, inverse @ shape, 'A⁻¹ applied to shape', inverse)
    fig.suptitle('Inverse by Gauss–Jordan elimination: [A | I] → [I | A⁻¹]', fontsize=13)
    finish(fig, filename)
