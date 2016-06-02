# -*- coding:utf-8 -*-

# This demo is based on O'reilly's website of Introduction to the t-SNE Algorithm

__author__ = "JOHNKYON"

# numpy and scipy imports
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# sklearn(Everyone is using this)
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# Code in sklearn 0.15.2
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
from sklearn.utils.extmath import _ravel

# Random state
RS = 20160510

# matplotlib for graphics
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# Import seaborn to make nice plots
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5, rc={"line.linewidth": 2.5})

# Generate an animation with matplotkib and moviepy
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

# load digits
digits = load_digits()
digits.data.shape

# print(digits['DESCR'])

nrows, ncols = 2, 5
plt.figure(figsize=(6, 3))
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i, ...])
    plt.xticks([]);
    plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('images/digits_generated.png', dpi=150)

# reorder the data points according to labels
X = np.vstack([digits.data[digits.target == i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target == i]
               for i in range(10)])

# digits_proj = TSNE(random_state=RS).fit_transform(X)


def scatter(x, colors):
    # color palette with seaborn
    palette = np.array(sns.color_palette("hls", 10))

    # Create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # Add labels for each digit
    txts = []
    for i in range(10):
        # Position of each label
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])
        txts.append(txt)

    return f, ax, sc, txts


# scatter(digits_proj, y)
# plt.savefig('images/digits_tsne-generated.png', dpi=120)


# compute similarity matrix
def _joint_probabilities_constant_sigma(D, sigma):
    # print D
    P = np.exp(-D ** 2 / 2 * sigma ** 2)
    P /= np.sum(P, axis=1)
    return P


# # Pairwise distances between all data points
# D = pairwise_distances(X, squared=True)
# # Similarity with constant sigma
# P_constant = _joint_probabilities_constant_sigma(D, .002)
# # Similarity with variable sigma
# P_binary = _joint_probabilities(D, 30., False)
# # The output of this function needs to be reshaped to a square matrix
# P_binary_s = squareform(P_binary)
#
# plt.figure(figsize=(12, 4))
# pal = sns.light_palette("blue", as_cmap=True)
#
# plt.subplot(131)
# plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
# plt.axis('off')
# plt.title("Distance matrix", fontdict={'fontsize': 16})
#
# plt.subplot(133)
# plt.imshow(P_binary_s[::10, ::10], interpolation='none', cmap=pal)
# plt.axis('off')
# plt.title('$p_{j|i}$ (variable $\sigma$)', fontdict={'fontsize': 16})
# plt.savefig('images/similarity-generated.png', dpi=120)


# This list will contain the positions of the map points at every iteration
positions = []


def _gradient_descent(objective, p0, it, n_iter, n_iter_check,
                      kwargs, n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=[],objective_error=None):
    # The documentation of this function can be found in scikit-learn's code.
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        # We save the current position.
        positions.append(p.copy())

        new_error, grad = objective(p, *args)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

    return p, error, i
sklearn.manifold.t_sne._gradient_descent = _gradient_descent

X_proj = TSNE(random_state=RS).fit_transform(X)

X_iter = np.dstack(position.reshape(-1, 2) for position in positions)

f, ax, sc, txts = scatter(X_iter[..., -1], y)


def make_frame_mpl(t):
    i = int(t*40)
    x = X_iter[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(10), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

npoints = 1000
plt.figure(figsize=(15, 4))
for i, D in enumerate((2, 5, 10)):
    # Normally distributed points.
    u = np.random.randn(npoints, D)
    # Now on the sphere.
    u /= norm(u, axis=1)[:, None]
    # Uniform radius.
    r = np.random.rand(npoints, 1)
    # Uniformly within the ball.
    points = u * r**(1./D)
    # Plot.
    ax = plt.subplot(1, 3, i+1)
    ax.set_xlabel('Ball radius')
    if i == 0:
        ax.set_ylabel('Distance from origin')
    ax.hist(norm(points, axis=1),
            bins=np.linspace(0., 1., 50))
    ax.set_title('D=%d' % D, loc='left')
plt.savefig('images/spheres-generated.png', dpi=100, bbox_inches='tight')