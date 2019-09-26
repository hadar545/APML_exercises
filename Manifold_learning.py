import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
import numpy.linalg as LA
import numpy.random as rd

###########################################
#                                         #
#   Code for Manifold Learning Exercise   #
#                                         #
###########################################
colors = ['gold', 'lightblue', 'magenta', 'navy', 'coral', 'teal', 'maroon', 'aquamarine', 'purple', 'lime', 'pink',
          'wheat']


def digits_example():
    """get MNIST dataset"""

    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    return data, labels


def swiss_roll_example():
    """ returns swiss roll's dataset """

    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)
    return X, color


def faces_example(path):
    """returns the faces dataset"""

    with open(path, 'rb') as f:
        X = pickle.load(f)
    return X


def plot_with_images(X, images, title, spot, image_num=25, single=False):
    """
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param spot: location to plot on
    :param image_num: Number of images to plot along with the scatter plot.
    :param single: switch for a single plot or multiple subplots
    """

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        if single:
            plt.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000, extent=(x0, x1, y0, y1))
            plt.tick_params(axis='both', which='major', labelsize=7)
        else:
            spot.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000, extent=(x0, x1, y0, y1))
            spot.tick_params(axis='both', which='major', labelsize=7)

    if single:
        plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)
        plt.title(title, fontsize=8, fontweight='bold')
    else:
        spot.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)
        spot.set_title(title, fontsize=8, fontweight='bold')


def MDS(X, d):
    """
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    """

    N = X.shape[0]
    H = np.eye(N, dtype=np.float16) - (1 / N) * np.ones((N, N))
    S = -0.5 * H @ X @ H
    vals, S_eigenvectors = np.linalg.eigh(S)
    sing_vals = np.sqrt(vals[-d:])
    eig_vecs = S_eigenvectors[:, -d:]
    return sing_vals * eig_vecs, vals


def get_weight_row(data, i_KNN, i):
    """calculates the values of single row of the W matrix in LLE algorithm and returns it"""
    A = data[i_KNN, :]
    Z = A - data[i, :]
    gram_matrix = Z @ Z.T
    w_vec = np.sum(LA.pinv(gram_matrix), axis=0)
    return w_vec / np.sum(w_vec)


def get_weight_matrix(data, KNN):
    """computes the W matrix for the LLE algorithm and returns it"""
    N = data.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        W[i, KNN[i, :]] = get_weight_row(data, KNN[i, :], i)
    return W


def LLE(X, d, k):
    """
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    """

    dists = euclidean_distances(X, X)
    N = dists.shape[0]
    KNN = np.argsort(dists, axis=1)[:, 1:k + 1]
    W = get_weight_matrix(X, KNN)
    double_M = (np.eye(N) - W).T @ (np.eye(N) - W)
    vals, M_eigenvectors = np.linalg.eigh(double_M)
    return M_eigenvectors[:, 1:d + 1]


def DiffusionMap(X, d, sigma, t):
    """
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    gram matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the gram matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    """
    N = X.shape[0]
    rs = np.arange(N)[np.newaxis].T

    # for the kernel similarity matrix
    dists = euclidean_distances(X, X)
    KNN_idx = np.argsort(dists, axis=1)
    KNN_dists = dists[rs, KNN_idx]
    heat_powers = -(KNN_dists ** 2) / sigma
    heat_kernel = np.exp(heat_powers)
    K = np.zeros((N, N))
    K[rs, KNN_idx] = heat_kernel

    # normalize the rows
    D = np.diagflat(np.sum(K, axis=1))
    A = LA.pinv(D) @ K

    # decompose A to form the embedded date at the right diffusion time
    vals, A_eigenvectors = np.linalg.eigh(A)
    sorted_indices = np.argsort(vals)[-(d + 1):-1]
    highest_vals = np.power(vals[sorted_indices], t)
    eig_vecs = A_eigenvectors[:, sorted_indices]
    return highest_vals * eig_vecs


def ScreePlot(sigma, N, P, d, show=True):
    """
    creates random dataset (low dim 'hiding' in high dim) and applies MDS for detection of the intrinsic dimension.
    plots the results if 'show' mode is on. Shows results as scree plots
    :param sigma: natural number between 1 and 9 describing the amount of noise
    :param N: the number of data points
    :param P: the high dimension
    """
    # create dataset in R_d
    intrinsic_data = rd.randint(0, 100, (N, d))
    high_data = np.concatenate((intrinsic_data, np.zeros((N, P - d))), axis=1)
    normal_rotation = rd.normal(0, 1, (P, P)) * 100
    Z = rd.normal(50, 100, (N, P))
    Q, _ = LA.qr(normal_rotation)
    X = high_data @ Q
    A = X + sigma * Z

    dists = euclidean_distances(A, A) ** 2
    _, eigvals = MDS(dists, d)
    if show:
        d_fit_plot(eigvals, len(eigvals), 'eigenvals for choosing the best d')
    else:
        return eigvals


def d_fit_plot(scores, max_d, title):
    """
    general bar plot function for fitting d given scores list
    :param scores: values to bar
    :param max_d: maximal value of d tested
    :param title: title of the plot
    """
    plt.bar(np.arange(max_d + 1)[1:], np.flip(scores), capstyle='butt', edgecolor='navy', alpha=0.5,
            color='#CC99CC', linestyle='--')
    plt.plot(np.arange(max_d + 1)[1:], np.flip(scores), '.r-', color='#FFCC00')
    plt.ylabel('Score')
    x_ticks = np.arange(max_d + 1)[1:]
    plt.xticks(x_ticks, x_ticks.astype(str))
    plt.title(title, fontsize=10, fontweight='bold')
    plt.show()


def noise_MDS_exploration():
    """explores the influence of varying noise levels over the performance of MDS algorithm. Dataset is randomly
    generates"""
    ncols = 3
    nrows = 3
    N = 10
    P = 15
    d = 3
    sigmaot = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 6, 10]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    x_ticks = np.arange(N + 1)[1:]
    for i in range(nrows):
        for j in range(ncols):
            eigvals = ScreePlot(sigmaot[i * nrows + j], N, P, d, False)
            ax[i, j].bar(np.arange(N + 1)[1:], np.flip(eigvals), capstyle='butt', edgecolor='navy', alpha=0.5,
                         color='#CC99CC', linestyle='--')
            ax[i, j].plot(np.arange(N + 1)[1:], np.flip(eigvals), '.r-', color='#FFCC00')
            ax[i, j].set_title('sigma = ' + str(sigmaot[i * nrows + j]), fontsize=8, fontweight='bold')
            ax[i, j].tick_params(axis='both', which='major', labelsize=7)
            ax[i, j].set_xticks(x_ticks)
            ax[i, j].set_facecolor('#E6E6E6')
    plt.suptitle('Exploring MDS results with varying noise rates\nvia scree plots of eigenvalues')
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.08, right=0.73, top=0.89, wspace=0.31, hspace=0.35)
    plt.show()


def dimension_exploration_scree():
    """
    exploring MDS's performance when changing the intrinsic dimension of a random generated data. creates multiple
    scree plots, one for each dimension checked, 4 overall
    """
    sigma = 0.05
    N = 10
    dims = [5, 10, 30, 60]
    fig, ax = plt.subplots(nrows=1, ncols=len(dims))
    x_ticks = np.arange(N + 1)[1:]
    for i in range(len(dims)):
        eigvals = ScreePlot(sigma, N, dims[i], 3, False)
        ax[i].bar(np.arange(N + 1)[1:], np.flip(eigvals), capstyle='butt', edgecolor='navy', alpha=0.5,
                  color='#CC99CC', linestyle='--')
        ax[i].plot(np.arange(N + 1)[1:], np.flip(eigvals), '.r-', color='#FFCC00')
        ax[i].set_title('Dimension = ' + str(dims[i]), fontsize=8, fontweight='bold')
        ax[i].set_xticks(x_ticks)
    plt.show()


def plot_MNIST_scatters(spot, labels, data, title, single=False):
    """ given a location in a figure, plots embedded MNIST data as scatters """
    c_num = np.unique(labels)
    # finds limits of figure
    scale = 1.5
    x_lim = np.max(np.absolute(data[:, 0]))
    y_lim = np.max(np.absolute(data[:, 1]))

    # scatters each digit separately
    for i in c_num:
        idx = labels == i
        label_data = data[idx, :]
        if single:
            plt.scatter(label_data[:, 0], label_data[:, 1], c=colors[i], s=5)
            plt.xlim((-x_lim * scale, x_lim * scale))
            plt.ylim((-y_lim * scale, y_lim * scale))
        else:
            spot.scatter(label_data[:, 0], label_data[:, 1], c=colors[i], s=5)
            spot.tick_params(axis='both', which='major', labelsize=7)
            spot.axis(np.array([-x_lim, x_lim, -y_lim, y_lim]) * scale)

    # title the plot
    if single:
        plt.title(title)
    else:
        spot.set_title(title, fontsize=8, fontweight='bold')


def MNIST_exploration():
    """
    explore how each manifold learning algorithm performs over the MNIST dataset. Explores how changes in
    hyperparameters can modify an algorithm's ability to discover the intrinsic dimension
    """
    X, labels = digits_example()
    N = X.shape[0]
    delta_matrix = euclidean_distances(X, X)

    # MDS graphs
    MDS_embedded, MDS_eigvals = MDS(delta_matrix, 2)
    plot_MNIST_scatters(None, labels, MDS_embedded, 'MDS over MNIST', True)
    plt.show()

    # LLE graphs
    ks = [7, 10, 15, 20, 30]
    # ks = [50, 150, 200]
    k_ncols = len(ks)
    _, ax = plt.subplots(nrows=1, ncols=k_ncols)
    for i in range(k_ncols):
        LLE_embbedded = LLE(X, 10, ks[i])
        plot_MNIST_scatters(ax[i], labels, LLE_embbedded, 'k = ' + str(ks[i]))
    plt.suptitle('LLE over MNIST dataset')
    plt.show()

    # Diffusion Maps Graphs
    sigma = 100
    sigmaot = [1, 10, 100, 1000]
    ts = [5, 10, 25, 50, 100]
    t = 20

    # exploring different sigma values
    dm_ncols = len(sigmaot)
    _, ax = plt.subplots(nrows=1, ncols=dm_ncols)
    for i in range(dm_ncols):
        DM_embedded = DiffusionMap(X, 2, sigmaot[i], t)
        plot_MNIST_scatters(ax[i], labels, DM_embedded, 'sigma = ' + str(sigmaot[i]))
    plt.suptitle('Diffusion Map over MNIST dataset')
    plt.show()

    # exploring different t values
    dm_ncols = len(ts)
    _, ax = plt.subplots(nrows=1, ncols=dm_ncols)
    for i in range(dm_ncols):
        DM_embedded1 = DiffusionMap(X, 2, sigma, ts[i])
        plot_MNIST_scatters(ax[i], labels, DM_embedded1, 't = ' + str(ts[i]))
    plt.suptitle('Diffusion Map over MNIST dataset')
    plt.show()


def plot_swiss_roll(spot, labels, data, title, single=False):
    """
    plots the swiss roll manifold created by one of manifold learning algorithms
    :param spot: location to plot in
    :param labels: true colors of the manifold
    :param data: embedded data
    :param title: title for the figure
    :param single: is this the only plot you wish to plot in this figure?
    """
    # figure out the plot's limits
    scale = 1.5
    x_lim = np.max(np.absolute(data[:, 0]))
    y_lim = np.max(np.absolute(data[:, 1]))

    # plot the data
    if single:
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=3.5, cmap='magma')
        plt.tick_params(axis='both', which='major', labelsize=7)
        plt.xlim((-x_lim * scale, x_lim * scale))
        plt.ylim((-y_lim * scale, y_lim * scale))
    else:
        spot.scatter(data[:, 0], data[:, 1], c=labels, s=3.5, cmap='magma')
        spot.tick_params(axis='both', which='major', labelsize=7)
        spot.axis(np.array([-x_lim, x_lim, -y_lim, y_lim]) * scale)

    # title the figure
    if single:
        plt.title(title)
    else:
        spot.set_title(title, fontsize=8, fontweight='bold')


def swiss_roll_exploration():
    """
    explore how each manifold learning algorithm performs over sklearn's swiss roll dataset. Explores how changes in
    hyperparameters can modify an algorithm's ability to discover the intrinsic dimension
    """
    X, labels = swiss_roll_example()
    delta_matrix = euclidean_distances(X, X) ** 2
    N = X.shape[0]

    # MDS Graphs
    MDS_embedded, MDS_eigvals = MDS(delta_matrix, 2)
    plot_swiss_roll(None, labels, MDS_embedded, 'MDS', True)
    plt.show()

    # LLE graphs
    ks = [5, 10, 20, 30]
    # ks = [50, 150, 200]
    k_ncols = len(ks)
    _, ax = plt.subplots(nrows=1, ncols=k_ncols)
    for i in range(k_ncols):
        LLE_embbedded = LLE(X, 2, ks[i])
        plot_swiss_roll(ax[i], labels, LLE_embbedded, 'k = ' + str(ks[i]))
    plt.suptitle('LLE applied over swiss roll dataset')
    plt.show()

    # Diffusion Maps Graphs
    sigmaot = [10, 15, 20]
    ts = [5, 10, 25, 50, 100]
    t = 20

    # exploring different sigma values
    dm_ncols = len(sigmaot)
    _, ax = plt.subplots(nrows=1, ncols=dm_ncols)
    for i in range(dm_ncols):
        DM_embedded = DiffusionMap(X, 2, sigmaot[i], t)
        plot_swiss_roll(ax[i], labels, DM_embedded, 'sigma = ' + str(sigmaot[i]))
    plt.suptitle('Diffusion Map algorithm applied over swiss roll dataset')
    plt.show()

    # exploring different t values
    dm_ncols = len(ts)
    sigmas = [7, 10, 15, 20]
    for sig in sigmas:
        _, ax = plt.subplots(nrows=1, ncols=dm_ncols)
        for i in range(dm_ncols):
            DM_embedded1 = DiffusionMap(X, 2, sig, ts[i])
            plot_swiss_roll(ax[i], labels, DM_embedded1, 't = ' + str(ts[i]))
        plt.suptitle('Diffusion Map algorithm applied over swiss roll dataset')
        plt.show()


def faces_exploration():
    """
    explore how each manifold learning algorithm performs over dataset of images portraing the face of a greek
    sculpture from various angles. Explores how changes in hyperparameters can modify an algorithm's ability to
    discover the intrinsic dimension
    """
    X = faces_example('faces.pickle')
    N = X.shape[0]
    
    # MDS Graphs
    delta_matrix = euclidean_distances(X, X) ** 2
    MDS_embedded, MDS_eigvals = MDS(delta_matrix, 2)
    plot_with_images(MDS_embedded, X, 'MDS', None, 100, True)
    plt.show()

    # LLE Graphs
    ks = [5, 10, 15, 25]
    # ks = [50, 150, 200]
    LLE_embbedded = LLE(X, 2, 15)
    plot_with_images(LLE_embbedded, X, 'LLE', None, 80, True)
    plt.show()

    k_ncols = len(ks)
    _, ax = plt.subplots(nrows=1, ncols=k_ncols)
    for i in range(k_ncols):
        LLE_embbedded = LLE(X, 2, ks[i])
        plot_with_images(LLE_embbedded, X, 'k = ' + str(ks[i]), ax[i], 60)
    plt.suptitle('LLE applied over Faces dataset')
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.54, right=0.95, top=0.9, wspace=0.2, hspace=0.2)
    plt.show()

    # Diffusion Maps Graphs
    sigmaot = [40, 50, 60, 70]
    t = 20

    # exploring different sigma values
    dm_ncols = len(sigmaot)
    _, ax = plt.subplots(nrows=1, ncols=dm_ncols)
    for i in range(dm_ncols):
        DM_embedded = DiffusionMap(X, 2, sigmaot[i], t)
        plot_with_images(DM_embedded, X, 'sigma = ' + str(sigmaot[i]), ax[i], 100)
    plt.suptitle('Diffusion Map algorithm applied over swiss roll dataset')
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.54, right=0.95, top=0.9, wspace=0.2, hspace=0.2)
    plt.show()


if __name__ == '__main__':
    noise_MDS_exploration()
    dimension_exploration_scree()
    MNIST_exploration()
    swiss_roll_exploration()
    faces_exploration()
