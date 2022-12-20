# Utilities for INFO 5604
# Fall 2022
# Lab 6 solution version

# Imports
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings 
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import mean_squared_error, silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.cluster._agglomerative import AgglomerativeClustering



def sigmoid(z):
    """
    Sigmoid (or logit) function. From a value to a probability.
    """
    return 1.0 / (1.0 + np.exp(-z))

def inv_sigmoid(p):
    """
    Inverse of the logit functon. From a probabiity to the associated value.
    """
    return - np.log((1/p) - 1)


def print_accuracies(classifier, X_train, y_train, X_test, y_test):
    """
    Print the training and test accuracy of a classifier. It also returns the two values
    in case they are needed for further processing.
    
    Arguments
    classifier
    X_train
    y_train
    X_test
    y_test
    
    """
    
    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)
    
    print(f'Train accuracy {train_acc:.3f}\tTest accuracy {test_acc:.3f}')
    
    return [train_acc, test_acc]

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """
    Plots 2D decision regions. From _Python Machine Learning 3rd ed._ by Raschka and Mirjalili
    """

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore", category=UserWarning, message=".*You passed a edgecolor/edgecolors.*")

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        alpha=0.8, 
                        color=colors[idx],
                        marker=markers[idx], 
                        label=cl, 
                        edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black',
                    alpha=1.0, linewidth=1,
                    marker='o', s=100, label='test set')  
        
def plot_learning_curve(train_fractions, train_scores, test_scores, score_name='Accuracy'):
    """
    Plots the learning curve from the results of the scikit-learn 'learning_curve' function.
    """

    train_dfs = [pd.DataFrame({'Fraction': frac, 'Data': 'Training', score_name: train_scores}) 
                       for frac, train_scores in zip(train_fractions, train_scores)]

    cv_dfs = [pd.DataFrame({'Fraction': frac, 'Data': 'Cross-validation', score_name: test_scores}) 
                       for frac, test_scores in zip(train_fractions, test_scores)]

    results_df = pd.concat(train_dfs + cv_dfs, axis = 0, ignore_index=True)
    sb.lineplot(data=results_df, x='Fraction', y=score_name, hue='Data', estimator='mean')
    
def print_errors(regressor, X_train, y_train, X_test, y_test):
    """
    Print the training and test MSE of a regressor. It also returns the two values
    in case they are needed for further processing.
    
    Arguments:
    regressor
    X_train
    y_train
    X_test
    y_test
    """
    
    train_error = mean_squared_error(y_train, regressor.predict(X_train))
    test_error  = mean_squared_error(y_test, regressor.predict(X_test))
    
    print(f'Train error {train_error:.3f}\tTest error {test_error:.3f}')
    
    return [train_error, test_error]

# Adapted from the scikit-learn documentation
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html?highlight=silhouette

def cluster_scoring(model, cluster_range, data, random_state=20221003, plot_silhouettes=False):
    db_vals = []
    ste_vals = []
    
    for n_clusters in cluster_range:
        
        # Initialize the clusterer with n_clusters value and a random generator seed. Seed is not needed for agglomerative
        if model != AgglomerativeClustering:
            clusterer = model(n_clusters=n_clusters, random_state=random_state)
        else:
            clusterer = model(n_clusters=n_clusters)
            
        cluster_labels = clusterer.fit_predict(data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        db_score = davies_bouldin_score(data, cluster_labels)
        ste_vals.append(silhouette_avg)
        db_vals.append(db_score)
        
        if plot_silhouettes:
        
            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots()

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 
            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title(f"The silhouette plot for {n_clusters} clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
    return pd.DataFrame({'Silhouette': ste_vals, 'Davies-Bouldin': db_vals, 'Clusters': cluster_range})

def plot_grid_search (grid_search, output_score='mean_test_score', output_label='Accuracy', x=None, hue=None, log_scale=False):
    params = grid_search.cv_results_['params']
    scores = grid_search.cv_results_[output_score]
    rows = [dict(param_dict, score=score) for param_dict, score in zip(params, scores)]
    cols = list(params[0].keys())
    cols.append(output_label)

    results_df = pd.DataFrame(rows)
    results_df.columns = cols
    
    if x is None:
        x = cols[0]
    if hue is None and len(cols) > 2:
        hue = cols[1]
        
    if hue is None:
        splot = sb.lineplot(data=results_df, x=x, y=output_label, marker='o')
    else: 
        splot = sb.lineplot(data=results_df, x=x, y=output_label, hue=hue, marker='o', palette='bright')
    
    if log_scale:
        splot.set(xscale='log')

    return results_df

def count_nz_cols(sp_mat):
    return np.sum(sp_mat.getnnz(axis=0)>0)

def word_weights(vectorizer, vectorizer_output, i):
    dense_vector = np.asarray(vectorizer_output[i].todense())[0]
    bool_vector =  dense_vector > 0
    weights = dense_vector[bool_vector]
    words = vectorizer.get_feature_names_out()[bool_vector]
    
    return {word: wt for word, wt in zip(words, weights)}


# From the scikit-learn documentation
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

def plot_top_words(model, feature_names, n_top_words, n_topics, title):
    fig, axes = plt.subplots(2, int(n_topics/2), figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
def print_top_words(mode, feature_names, n_top_words, topic_id):
    topic = mode.components_[topic_id]
    top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]
    
    for feature, wt in zip([top_features, weights]):
        print(f'{feature}: {wt:0.3f}')

def plot_history(h):
    history_df = pd.DataFrame(h.history)
    history_df = history_df.reset_index()
    history_df_long = history_df.melt(id_vars='index', value_vars=['loss', 'val_loss'])
    history_df_long.columns = ['Epoch', 'Loss Type', 'Loss']

    sb.lineplot(data=history_df_long, x='Epoch', y='Loss', hue='Loss Type')
    plt.legend(title='Loss Type', labels=['Training Loss', 'Validation Loss'])

