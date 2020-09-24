
from kmodes.kprototypes import KPrototypes

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import skfuzzy as fuzz

from sklearn.cluster import AgglomerativeClustering, KMeans


from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler


def clusterization(data, clusters, method):
    if method is 'kmeans':
        model = KMeans(n_clusters=clusters, init='random', algorithm='full')
        model.fit(data)
        clustering_labels = model.predict(data)
    elif method is 'agglomerative':
        linkage = ('ward', 'average', 'complete', 'single')
        model = AgglomerativeClustering(linkage=linkage[0], n_clusters=clusters)
        model.fit(data)
        clustering_labels = model.labels_
    elif method is 'fuzzy':
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, clusters, 2, error=0.005, maxiter=10000, init=None)
        clustering_labels = np.argmax(u, axis=0)
    elif method is 'kprototypes':
        clustering_labels = KPrototypes(n_clusters=clusters, init='random', gamma=0.1, n_init=1).fit_predict(data, categorical=list([8]))
    else:
        print(" The supported methods are: kmeans, agglomerative, fuzzy ...")
    return clustering_labels


def clusterization_by_number_of_clusters(data, labels_geo, num_clusters, method):
    labels_geo = np.expand_dims(labels_geo, axis=-1)
    results = []
    for n_clusters in range(num_clusters, 1, -1):
        predictions = np.expand_dims(clusterization(data, n_clusters, method=method), axis=-1)
        results.append(predictions)

    results = np.concatenate(results, axis=-1)
    results = np.concatenate((labels_geo, results), axis=-1)
    colummns_names = np.concatenate((np.array(['Geology']), np.arange(num_clusters, 1, -1)))

    results_df = pd.DataFrame(results, columns=colummns_names)
    return results_df


def organize_results_by_geology(results_df, labels_geo, num_clusters):
    labels_geo = np.expand_dims(labels_geo, axis=-1)
    geo_classes = np.unique(labels_geo)
    num_classes = len(geo_classes)

    results_by_geology_total = []
    cluster_names = []
    for n_clusters in range(num_clusters, 1, -1):
        cluster_names.append(np.arange(n_clusters))
        results_by_geology = np.zeros((num_classes, n_clusters))
        labels_by_clustering = results_df.iloc[:, num_clusters-n_clusters+1:num_clusters-n_clusters+2].to_numpy().astype('float')

        results_by_clustering = np.concatenate((labels_geo, labels_by_clustering), axis=-1)
        results_by_clustering_df = pd.DataFrame(results_by_clustering, columns=['Geology', 'clustering'])

        for geo_class in range(num_classes):
            for cluster in range(n_clusters):
                freq = np.sum(results_by_clustering_df[results_by_clustering_df['Geology']==geo_classes[geo_class]]['clustering']==cluster)
                results_by_geology[geo_class, cluster] = freq
        results_by_geology_total.append(results_by_geology)

    cluster_names = np.concatenate(cluster_names)
    cluster_names = np.concatenate((np.array(['Geology']), cluster_names))
    results_by_geology_total = np.concatenate(results_by_geology_total, axis=-1)
    geo_classes = np.expand_dims(geo_classes, axis=-1)
    results_by_geology_total = np.concatenate((geo_classes, results_by_geology_total), axis=-1)
    results_by_geology_total_df = pd.DataFrame(results_by_geology_total, columns=cluster_names)
#     print(results_by_geology_total.shape)
    return results_by_geology_total_df


def compute_correlation_matrices(results_by_geology, geo_classes, num_clusters, only_arenito=True):
    num_classes = len(geo_classes)
    # if true, only arenito samples are considered 
    if only_arenito:
        # ther first 26 rows correspond to arenito samples
        num_classes = 26
        geo_classes = geo_classes[:num_classes]

    # reading results organized by geology classes
    left_idx = 1
    corr_matrices = np.zeros((num_classes * num_classes, num_clusters-1))
    for n_clusters in range(num_clusters, 1, -1):
    # print("Correlation matrix using clusters: ", n_clusters)
        results_by_cluster = results_by_geology.iloc[:, left_idx:left_idx+n_clusters].to_numpy().astype('float')
        left_idx = left_idx + n_clusters

        # filtering cinsidered classes
        results_by_cluster = results_by_cluster[:num_classes]

        # compute unitary representation vectors
        results_by_cluster_norm = results_by_cluster/np.linalg.norm(results_by_cluster, axis=1, keepdims=True)

        # compute correlation matrix by experiment
        corr_matrix = np.dot(results_by_cluster_norm, results_by_cluster_norm.T)
        corr_matrices[:, num_clusters-n_clusters] = corr_matrix.ravel()

    return corr_matrices


def plot_correlation_matrices(corr_matrices, geo_classes, num_clusters, name=None):

    # Generate a mask for the upper triangle
    num_classes = np.sqrt(corr_matrices.shape[0]).astype('int64')
    mask = np.zeros((num_classes, num_classes), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # setting parameters for ploting
    sns.set(style="white")
    for n_clusters in range(num_clusters, 1, -1):
        print("Correlation matrix using clusters: ", n_clusters)
        corr_matrix = corr_matrices[:, num_clusters - n_clusters].reshape(num_classes, num_classes)
        f, ax = plt.subplots(figsize=(30, 30))
        sns.heatmap(corr_matrix,
                    mask=mask,
                    square=True,
                    linewidths=.5,
                    cmap='coolwarm',
                    cbar=False,
                    cbar_kws={'shrink': .4, 'ticks': [0, .25, 0.5, .75, 1]},
                    vmin=0,
                    vmax=1,
                    annot=True,
                    annot_kws={'size': 20})
        ax.set_yticklabels(geo_classes, rotation=0, fontsize=18)
        ax.set_xticklabels(geo_classes, rotation=90, fontsize=18)
        sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
        plt.tight_layout()

        if name:
            plt.savefig(name + 'correlation_matrix_' + str(n_clusters) + '.png', dpi=150)
        plt.show()

    return


def compute_indexes_for_high_correlated_classes(corr_matrices, num_clusters, th=0.90):
    # Generate a mask for the upper triangle
    num_classes = np.sqrt(corr_matrices.shape[0]).astype('int64')
    mask = np.zeros((num_classes, num_classes), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    index_high_correlated = []
    high_correlated_classes_by_exp = np.zeros((num_classes * num_classes, num_clusters-1), dtype='uint8')
    for n_clusters in range(num_clusters, 1, -1):
        # print("Correlation matrix using clusters: ", n_clusters)
        corr_matrix = corr_matrices[:, num_clusters-n_clusters].reshape(num_classes, num_classes)

        # Compute index for classes high correlated
        h_idx = np.arange(corr_matrix.size)
        h_idx = h_idx.reshape(corr_matrix.shape)

        corr_matrix[mask] = 0  # Consider inferior triangular matrix values
        h_idx[mask] = 0
        h_idx[corr_matrix < th] = 0
        h_idx = h_idx[h_idx != 0]
        index_high_correlated.append(h_idx)
        high_correlated_classes_by_exp = fill_high_correlated_matrix(high_correlated_classes_by_exp, h_idx, n_clusters)
    index_high_correlated = np.concatenate(index_high_correlated)

    return high_correlated_classes_by_exp, index_high_correlated


def fill_high_correlated_matrix(high_correlated_by_cluster, idx, cluster):
    for group in idx:
        high_correlated_by_cluster[group, cluster - 2] = 255
    return high_correlated_by_cluster


def plot_fzi_distribution_by_clustering(results_total_df, fzi, cluster_exps_list, num_clusters, name=None):
    fzi = np.expand_dims(fzi, axis=-1)
    # results_total = np.expand_dims(results_total_df.to_numpy()[:, 1:].astype('float'), axis=-1)
    for n_clusters in cluster_exps_list:
        clustering_by_exp = np.expand_dims(results_total_df[str(n_clusters)].astype('float'), axis=-1)
        fzi_by_cluster = np.concatenate((fzi, clustering_by_exp), axis=-1)
        fzi_by_cluster_df = pd.DataFrame(fzi_by_cluster, columns=['FZI', 'clustering_labels'])

        fzi_by_clusters = []
        for cluster_index in range(len(np.unique(clustering_by_exp))):
            fzi_by_clusters.append(fzi_by_cluster_df['FZI'][np.squeeze(clustering_by_exp==cluster_index)].values)
        f, ax = plt.subplots(figsize=(6, 5))
        sns.set(style="whitegrid")
        sns.boxplot(data=fzi_by_clusters, showfliers=False)
        ax.set_yticklabels(np.arange(0, fzi.max()), fontsize=11)
        ax.set_xticklabels(np.arange(1, n_clusters + 1), fontsize=11)
        plt.xlabel('clusters')
        plt.ylabel('FZI')
        if name:
            plt.savefig(name + 'fzi_vs_num_clusters_' + str(n_clusters) + '.png', dpi=150)
        plt.show()


def compute_most_freq_classes(high_correlated_idx, geo_classes, freq=4, only_arenito=True, name=None):
    num_classes = len(geo_classes)
    groups = np.unique(high_correlated_idx)
    count_groups = []
    if only_arenito:
        num_classes = 26
    for group in groups:
        row = (np.int64)(group / num_classes)
        column = group % num_classes
        row, column
        group_name = geo_classes[row] + '<-->' + geo_classes[column]
        repeated = (high_correlated_idx == group).sum()
        if repeated > freq:
            count_groups.append([group_name, repeated, group])

    count_groups_df = pd.DataFrame(count_groups, columns=['group_name', 'occurences', 'group'])
    count_groups_df.sort_values(by=['occurences'], inplace=True)
    return count_groups_df


def diagram_bars_freq_classes(count_groups_df, name=None):
    num_ele = len(count_groups_df)
    count_groups_df.sort_values(by=['occurences'], ascending=True, inplace=True)
#     print(count_groups_df)
    for i in range(0, len(count_groups_df) // num_ele):
        plt.figure(figsize=(10, 15))
        plt.barh(range(num_ele), count_groups_df['occurences'].values[i*num_ele:(i+1)*num_ele], height=0.8, tick_label=count_groups_df['group_name'].values[i*num_ele:(i+1)*num_ele])
        plt.xlim(0, 27)
        plt.tight_layout()
        if name:
            plt.savefig(name + 'diagram_bars_' + str(i) + '.png', dpi=150)
        plt.show()


def heatmap_freq_classes(count_groups_df, high_correlated, name=None):
    count_groups_df.sort_values(by=['occurences'], ascending=False, inplace=True)
    f, ax = plt.subplots(figsize=(10, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(high_correlated[count_groups_df['group']],
                cmap=cmap, center=0, cbar=False,
                xticklabels=range(2, 28), yticklabels=count_groups_df['group_name'],
                square=True, linewidths=.5)
    plt.tight_layout()
    if name:
        plt.savefig(name + 'heatmaps' + '.png', dpi=150)
    plt.show()


# preoprocessing data

def interpolate_data_by_well(data_df, plot=False):
    # Sort data by Well and Depth
    df = data_df
    df.sort_values(by=['Well', 'Depth'], inplace=True)

    # Interpolate data
    DT, GR, NPHI, RHOB = [], [], [], []

    for well in np.unique(df['Well']):
        DT.append(df[df['Well'] == well]['DT'].interpolate().values)
        GR.append(df[df['Well'] == well]['GR'].interpolate().values)
        NPHI.append(df[df['Well'] == well]['NPHI'].interpolate().values)
        RHOB.append(df[df['Well'] == well]['RHOB'].interpolate().values)
        if plot:
            plt.plot(df[df['Well'] == well]['DT'].interpolate().values)
            plt.show()

    dt = np.concatenate(DT)
    gr = np.concatenate(GR)
    nphi = np.concatenate(NPHI)
    rhob = np.concatenate(RHOB)

    df['F_DT'] = dt
    df['F_GR'] = gr
    df['F_NPHI'] = nphi
    df['F_RHOB'] = rhob
    return df


def normalize_data_by_well(df, method='standard'):
    # Normalize data per Well
    # scaler = PowerTransformer()
    # StandardScaler()
    # MinMaxScaler((-1, 1))
    scaled_data = []
    for well in np.unique(df['Well']):
        if method is 'standard':
            scaler = PowerTransformer()
        data = df[df['Well'] == well][['F_DT', 'F_GR', 'F_NPHI', 'F_RHOB']]
        data_scaled_by_well = scaler.fit_transform(data)
        scaled_data.append(data_scaled_by_well)

    scaled_data = np.concatenate(scaled_data)

    df['S_DT'] = scaled_data[:, 0]
    df['S_GR'] = scaled_data[:, 1]
    df['S_NPHI'] = scaled_data[:, 2]
    df['S_RHOB'] = scaled_data[:, 3]
    return df


def mean_by_window(dataset, history_size):
    dataset = np.pad(dataset, ((history_size // 2, history_size // 2), (0, 0)), 'reflect') # paddind dataset
    data = []
    for i in range(history_size // 2, len(dataset) - history_size // 2):
        indices = range(i - history_size // 2, i + history_size // 2 + 1)
        data.append(dataset[indices].mean(axis=0))

    return np.array(data)


def windowing_data_center(dataset, history_size):
    dataset = np.pad(dataset, ((history_size // 2, history_size // 2), (0, 0)), 'reflect') # paddind dataset
    data = []
    for i in range(history_size // 2, len(dataset) - history_size // 2):

        indices = range(i - history_size // 2, i + history_size // 2 + 1)
#         print(indices)
        data.append(np.reshape(dataset[indices], (history_size, dataset.shape[1])))

    return np.array(data)


def scaling_data(df, sismic_data_labeled, geo_data_labeled):
    # scaler = StandardScaler()
    scaler = PowerTransformer()
    sismic_data_scaled = scaler.fit_transform(sismic_data_labeled)
    scaler = PowerTransformer()
    # scaler = StandardScaler()
    geo_data_scaled = scaler.fit_transform(geo_data_labeled)
    data_scaled = np.concatenate((sismic_data_scaled, geo_data_scaled), axis=-1)

    # Determine index for Arenito samples and assing weigths
    weight = np.mean(data_scaled.max(axis=0) - data_scaled.min(axis=0)) / 2
    isarenito = []
    for i in range(len(df['Geology'])):
        if 'A' in str(df['Geology'].iloc[i]):
            isarenito.append(weight)
        else:
            isarenito.append(-weight)
    isarenito = np.array(isarenito)
    isarenito = np.expand_dims(isarenito, axis=-1)
    data2evaluate = np.concatenate((data_scaled, isarenito), axis=-1)
    df_scaled = pd.DataFrame(data2evaluate, columns=['DT', 'GR', 'NPHI', 'RHOB', 'Permeabilidade', 'Porosidade', 'RQI', 'FZI', 'isArenito'])
    return df_scaled


    # trn_samples = np.concatenate((sismic_data_labeled, geo_data_labeled), axis=-1)
    # scaler = MinMaxScaler((-1, 1))
    # data_scaled_min_max = scaler.fit_transform(trn_samples)



def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
