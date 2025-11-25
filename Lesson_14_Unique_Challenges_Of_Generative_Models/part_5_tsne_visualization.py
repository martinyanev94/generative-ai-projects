from sklearn.manifold import TSNE

def visualize(data):
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.show()
