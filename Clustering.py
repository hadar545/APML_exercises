import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import copy

colors = ['gold', 'lightblue', 'magenta', 'navy', 'coral', 'teal', 'maroon', 'aquamarine', 'purple', 'lime', 'pink',
          'wheat']


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    return circles


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle'):
    """
    an example function for loading the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.

    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    return euclidean_distances(X, Y)


def euclidean_centroid(partial_X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    return np.mean(partial_X, axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    N, D = X.shape
    mu_1 = np.random.randint(0, high=N, size=1)[0]
    mu = X[mu_1, :]
    for i in range(k - 1):
        dists = metric(X, mu)
        nearest_centers_dists = np.min(dists, axis=1)
        nearest_centers_dists = np.power(nearest_centers_dists, 2)
        probabilities = nearest_centers_dists / np.sum(nearest_centers_dists)
        mu_i = np.random.choice(np.arange(N), 1, p=probabilities)[0]
        mu = np.vstack((mu, X[mu_i, :]))
    return np.matrix(mu)


def culc_objective(X, clusters, mu, k, metric):
    """
    calculate and return the value of the objective resulted from a specific clustering of the data
    :param X: dataset
    :param clusters: a list with the clusters info
    :param mu: centers
    :param k: number of clusters
    :param metric: distances metric
    """
    score = 0
    for j in range(k):
        idx = clusters == j
        if np.sum(idx):
            cluster_data = X[idx, :]
            j_dists = metric(cluster_data, mu[j])
            score += np.sum(j_dists ** 2)
    return score


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    N, D = X.shape
    clusters_history = np.zeros(N)
    mu_history = np.zeros((k, D))
    # save each iteration's data
    clusters_info = []
    mu_info = []
    objective_vals = []
    # run until convergence
    for i in range(iterations):
        mu = init(X, k, metric)
        while True:
            dists = metric(X, mu)
            clusters = np.argmin(dists, axis=1)
            for j in range(k):
                idx = clusters == j
                if np.sum(idx):
                    cluster_data = X[idx, :]
                    mu[j, :] = center(cluster_data)
            if np.allclose(clusters_history, clusters) and np.allclose(mu_history, mu):
                break
            clusters_history = clusters
            mu_history = mu
        clusters_info.append(clusters)
        mu_info.append(mu)
        # calculate the objective for this iteration
        iter_objective = culc_objective(X, clusters, mu, k, metric)
        objective_vals.append(iter_objective)
    best_iter = np.argmin(objective_vals)
    print(objective_vals[best_iter])

    return clusters_info[best_iter], mu_info[best_iter], np.min(objective_vals)


def draw_distances_corr(W_matrix, clustering):
    """
    plotts a heatmap of w_matrix to observe topological structures in the data
    :param W_matrix: similarity matrix
    :param clustering: clustering info
    """
    # shuffle then plot
    shuffled_w = copy.deepcopy(W_matrix)
    np.random.shuffle(shuffled_w)
    np.random.shuffle(shuffled_w.T)
    # correlation = np.corrcoef(shuffled_w)
    plt.figure()
    plt.imshow(shuffled_w, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()
    # order then plot
    idx = np.argsort(clustering)
    dists = W_matrix[idx, :]
    dists = dists[:, idx]
    # correlation = np.corrcoef(dists)
    plt.figure()
    plt.imshow(dists, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    powered_dists = np.power(X, 2)
    W_matrix = np.exp(-powered_dists / (2 * (sigma ** 2)))
    return W_matrix


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    N = X.shape[0]
    sorted_neigh = np.argsort(X, axis=1)[:, :m]
    row, _ = np.indices((N, m))
    mnn_adj = np.zeros((N, N))
    mnn_adj[row, sorted_neigh] = 1
    return np.logical_or(mnn_adj, mnn_adj.T)


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    N_n, D_ = X.shape
    dists = euclidean_distances(X, X)
    W_matrix = similarity(dists, similarity_param)
    sqrt_D = np.diagflat(np.power(np.sum(W_matrix, axis=1), -0.5))
    plot_dists_in_pairs_hist(dists)
    L_matrix = np.identity(N_n) - np.dot(np.dot(sqrt_D, W_matrix), sqrt_D)
    vals, L_eigenvectors = np.linalg.eigh(L_matrix)
    idx = np.argsort(vals)
    lowest_e_vecs = L_eigenvectors[:, idx[:k]]
    lowest_e_vecs = preprocessing.normalize(lowest_e_vecs, axis=1, norm='l2')
    k_fit_plot(vals[:15], 15, 'k Fit for K-means Using Eigen Gap')
    clustering, _1, _2 = kmeans(np.matrix(lowest_e_vecs), k)
    # draw_distances_corr(W_matrix, clustering)
    return clustering, _1, _2


def elbow(max_k, X):
    """
    implenetation of the elbow method for kmeans
    :param max_k: maximum value of k
    :param X: dataset
    """
    scores = []
    for k in range(1, max_k + 1):
        _, _, curr_score = kmeans(X, k)
        scores.append(curr_score)
    k_fit_plot(scores, max_k, 'K Fit for K-means using Elbow Method')


def run_Silhouette(X, max_k):
    """
    implementation of the silhouette method for finding the best k
    :param X: dataset
    :param max_k: maximal value of k to test
    """
    s_scores = silhouette(X, max_k)
    k_fit_plot(s_scores, max_k, 'K Fit for K-means Using Silhouette Method')


def k_fit_plot(scores, max_k, title):
    """
    general bar plot function for fitting k
    :param scores: values to bar
    :param max_k: maximal value of k tested
    :param title: title of the plot
    """
    plt.bar(np.arange(max_k + 1)[1:], scores, capstyle='butt', edgecolor='navy', alpha=0.5,
            color='turquoise', linestyle='--')
    plt.ylabel('Score')
    x_ticks = np.arange(max_k + 1)[1:]
    plt.xticks(x_ticks, x_ticks.astype(str))
    plt.title(title, fontsize=10, fontweight='bold')
    plt.show()


def silhouette(X, max_k):
    """
    inner function of run_silhouette which calculates the resulted scores for all the ks
    :param X: dataset
    :param max_k: maximal value of k to test
    :return: the silhouette scores
    """
    N = X.shape[0]
    dists = euclidean_distances(X, X)
    As = np.zeros((N, max_k))
    Bs = np.zeros((N, max_k))
    for k in range(2, max_k + 2):
        # clusters, mu, _ = kmeans(X, k)
        clusters, mu, _ = kmeans(X, k)
        cluster_dists = np.zeros((N, k))
        for j in range(k):
            if np.sum(clusters == j):
                C_dists = dists[clusters == j, :]
                cluster_dists[:, j] = np.sum(C_dists, axis=0) / C_dists.shape[0]
                C_dists = C_dists[:, clusters == j]
                a_j_vals = np.sum(C_dists, axis=1) / (C_dists.shape[0] - 1)
                np.put(As[:, k - 2], np.where(clusters == j)[0], a_j_vals)
        idx = np.arange(N)
        cluster_dists[idx, clusters] = np.inf
        Bs[:, k - 2] = np.min(cluster_dists, axis=1)
    S_vals = np.divide(np.subtract(Bs, As), np.maximum(As, Bs))
    return np.average(S_vals, axis=0)


def plot_spectral_over_apml(k, sim_param, w_func, path='APML_pic.pickle'):
    """
    applying spectral clustering over apml dataset and plotting the clustering
    :param k: desitred number of clusters
    :param sim_param: similarity parameter
    :param w_func: similarity function
    :param path: path of spectral dataset
    """
    with open(path, 'rb') as f:
        apml = pickle.load(f)
    clusters, mu, _ = spectral(apml, k, sim_param, w_func)
    for i in range(k):
        idx = clusters == i
        data = apml[idx, :]
        plt.scatter(data[:, 0], data[:, 1], s=2, c=colors[i])
    plt.show()


def plot_spectral_over_circles(k, sim_param, w_func):
    """
    runs spectral algorithm over circles dataset and plots the clustering
    :param k: number of clusters
    :param sim_param: similarity parameter
    :param w_func: desired similarity function
    """
    circles = circles_example().T
    clusters, mu, _ = spectral(circles, k, sim_param, w_func)
    for i in range(k):
        idx = clusters == i
        data = circles[idx, :]
        plt.scatter(np.array(data[:, 0].T)[0], np.array(data[:, 1].T)[0], s=2, c=colors[i])
    plt.show()


def plot_kmeans_over_apml(k, iter_num, path='APML_pic.pickle'):
    """
    clusters apml data using kmeans algorithm and plot the clustering
    :param k: the number of desired clusters
    :param iter_num: number of iteraions
    :param path: path of APML data
    """
    with open(path, 'rb') as f:
        apml = pickle.load(f)
    clusters, mu, _ = kmeans(np.matrix(apml), k, iter_num)
    for i in range(k):
        idx = clusters == i
        data = apml[idx, :]
        plt.scatter(data[:, 0], data[:, 1], s=2, c=colors[i])
    plt.show()


def plot_kmeans_over_circles(k, iter_num):
    """
    runs kmeans over the circles dataset and plots the clustering
    :param k: the desired number of clusters
    :param iter_num: number of iterations
    """
    circles = circles_example().T
    clusters, mu, _ = kmeans(circles, k, iter_num)
    for i in range(k):
        idx = clusters == i
        data = circles[idx, :]
        plt.scatter(np.array(data[:, 0].T)[0], np.array(data[:, 1].T)[0], s=2, c=colors[i])
    plt.show()


def microarray_clustering_graphs(k):
    """
    explors the mivroarray gene expression dataset by creating several plots
    :param k: the desired number of clusters
    """
    # creating distances matrix
    data = np.matrix(microarray_exploration())
    clusters, _, _ = kmeans(data, k)
    idxes = np.argsort(clusters)
    data_by_clusters = data[idxes, :]
    plt.figure()
    plt.imshow(data_by_clusters, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()

    # visualiza each cluster in a heatmap by itself
    for i in range(k):
        ix = clusters == i
        cluster_data = data[ix, :]
        plt.imshow(cluster_data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
        plt.colorbar()
        plt.show()

    # visualize the number of genes at each cluster
    w_init = dict(zip(*np.unique(clusters, return_counts=True)))
    w_values = np.sort(list(w_init.values()))
    plt.bar(np.arange(k + 1)[1:], w_values, capstyle='butt', edgecolor='navy', alpha=0.5,
            color='turquoise', linestyle='--')
    plt.ylabel('Score')
    x_ticks = np.arange(k + 1)[1:]
    plt.xticks(x_ticks, x_ticks.astype(str))
    plt.title("Number of Genes in Each Cluster", fontsize=10, fontweight='bold')
    plt.show()


def plot_tsne_pce(Y_tsne, Y_pca, labels1):
    """
    creates two plots corresponding to two compression methods
    :param Y_tsne: embedded data for tSNE
    :param Y_pca: embedded data for PCA
    :param labels1: the true taggs of each datapoint
    """
    fig, ax = plt.subplots(nrows=2, ncols=1)
    c_num = np.unique(labels1)
    for i in c_num:
        relevant_y_tsne = Y_tsne[np.where(labels1 == i)]
        relevant_y_pca = Y_pca[np.where(labels1 == i)]
        ax[0].scatter(relevant_y_tsne[:, 0], relevant_y_tsne[:, 1], c=colors[i], s=3.5, label=str(i))
        ax[1].scatter(relevant_y_pca[:, 0], relevant_y_pca[:, 1], c=colors[i], s=3.5, label=str(i))
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Compressed Data Using PCA')
    ax[1].set_ylabel('PC2')
    ax[1].set_xlabel('PC1')
    ax[0].set_title('Compressed Data Using t-SNE')
    ax[0].set_ylabel('tSNE1')
    ax[0].set_xlabel('tSNE2')
    plt.show()


def compare_compression_methods():
    """
    loads MNIST dataset creates PCA ant TSNE plots
    """
    # dataset that tsne can compress and pca fails
    X, labels = datasets.load_digits(return_X_y=True)
    X /= 255

    # tSNE_exploration(X, labels)
    tSNE_exploration(X, labels)


def tSNE_exploration(X, labels):
    """
    creates two plots - PCA and tSNE of a dataset
    :param X: dataset
    :param labels: the true tags of each datapoint
    """
    # tSNE
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Y_tsne = tsne.fit_transform(X)
    # PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    Y_pca = pca.transform(X)
    # plot the two figures
    plot_tsne_pce(Y_tsne, Y_pca, labels)


def plot_objective_for_spectral(data, k):
    """
    explors the effects of different sigme (or m) parantetrs over the objective acore of the spectral clustering
    :param data: dataset
    :param k: number of clusters
    """
    sigma_params = [0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1, 1.5, 3, 7, 15]
    m_params = [2, 3, 4, 5, 6, 7, 8, 9, 20, 50, 100, 500, 1000]
    sig_len = len(sigma_params)
    m_len = len(m_params)
    sigma_objectives = []
    m_objectives = []
    for param in sigma_params:
        clustering, mus, obj = spectral(data, k, param, similarity=gaussian_kernel)
        sigma_objectives.append(obj)
    for param in m_params:
        clustering, mus, obj = spectral(data, k, param, similarity=mnn)
        m_objectives.append(obj)

    plt.bar(np.arange(sig_len + 1)[1:], sigma_objectives, capstyle='butt', edgecolor='navy', alpha=0.5,
            color='turquoise', linestyle='--')
    plt.ylabel('Score')
    plt.xticks(np.arange(sig_len + 1)[1:], np.array(sigma_params).astype(str))
    plt.title('exploring different values of\n sigma for spectral clustring', fontsize=10, fontweight='bold')
    plt.show()

    plt.bar(np.arange(m_len + 1)[1:], m_objectives, capstyle='butt', edgecolor='navy', alpha=0.5,
            color='turquoise', linestyle='--')
    plt.ylabel('Score')
    plt.xticks(np.arange(m_len + 1)[1:], np.array(m_params).astype(str))
    plt.title('exploring diffeerent values of\n m neighbors for spectral clustering', fontsize=10, fontweight='bold')
    plt.show()


def embedd_synethetic_data():
    """
    creates synthetic data and embedds it
    """
    data, labels = datasets.make_blobs(n_samples=1000, n_features=25, centers=10)
    tSNE_exploration(data, labels)


def plot_dists_in_pairs_hist(dists):
    """
    plotts a histogram of the distances in paris of a dataset
    :param dists: dists matrix of a dataset points
    """
    dists_list = np.sort(np.triu(dists).flatten())
    dists_list = dists_list[np.where(dists_list > 0)]
    plt.hist(dists_list, bins=50, color=colors[4], range=[0, np.max(dists_list)])
    plt.show()


if __name__ == '__main__':
    # plot_kmeans_over_apml(10, 10)
    # plot_kmeans_over_apml(12, 10)
    # plot_kmeans_over_circles(5, 10)
    # compare_compression_methods()
    # embedd_synethetic_data()
    with open('APML_pic.pickle', 'rb') as f:
        apml = pickle.load(f)
    # plot_objective_for_spectral(apml, 9)
    # plot_spectral_over_apml(9, 7, gaussian_kernel)
    data = np.matrix(microarray_exploration())
    # spectral(data,12,3)
    circles = circles_example().T
    plot_spectral_over_apml(9, 9, gaussian_kernel)
    # plot_spectral_over_apml(6, 9, gaussian_kernel)
    # plot_spectral_over_apml(10, 9, gaussian_kernel)
    # plot_spectral_over_apml(9, 15, mnn)
    # plot_spectral_over_circles(5, 0.1, gaussian_kernel)
    # plot_spectral_over_circles(3, 6, mnn)
    # plot_spectral_over_circles(4,10, mnn)
    # plot_spectral_over_circles(10, 3, mnn)
    # run_Silhouette(data,15)
    # run_Silhouette(np.matrix(apml), 12)
    # run_Silhouette(circles, 7)
    # elbow(12, data)
    # elbow(12, np.matrix(apml))
    # kmeans(circles, 4, iterations=10)
    # microarray_clustering_graphs(3)
    # spectral(circles, 4, 6, mnn)
