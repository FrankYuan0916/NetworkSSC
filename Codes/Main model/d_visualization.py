from sklearn.manifold import TSNE
import seaborn as sns
from a_preData import *


def tsneData(network_result, ssc_result, adaptive_result, label):
    pca = PCA(n_components=50, random_state=42)
    pca_result_network = pca.fit_transform(network_result)
    pca_result_ssc = pca.fit_transform(ssc_result)
    pca_result_adaptive = pca.fit_transform(adaptive_result)
    tsne = TSNE(n_components=2, random_state=42)
    network_tsne = tsne.fit_transform(pca_result_network)
    ssc_tsne = tsne.fit_transform(pca_result_ssc)
    adaptive_tsne = tsne.fit_transform(pca_result_adaptive)

    # fig.add_subplot(121)
    df = pd.DataFrame()
    df["y"] = label
    # df["label"] = model.pre_data.label
    df["network x"] = network_tsne[:,0]
    df["network y"] = network_tsne[:,1]
    df["ssc x"] = ssc_tsne[:,0]
    df["ssc y"] = ssc_tsne[:,1]    
    df["adaptive x"] = adaptive_tsne[:,0]
    df["adaptive y"] = adaptive_tsne[:,1]    
    return df


def tsnePlot(network_W, ssc_W, adaptive_W, label, dataname):
    df = tsneData(network_W, ssc_W, adaptive_W, label)
    plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#997273","#3A84E6","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    # # adata.obs['network'] = adata.obs['network'].astype('str')
    cluster_uniq = np.sort(np.unique(label))
    col_map = {}
    for i in range(len(cluster_uniq)):
        col_map[cluster_uniq[i]] = plot_color[i]
    df = df.sort_values(by=['y'])

    df['y'] = df['y'].astype('str')

    fig, axes = plt.subplots(1,3,figsize=(18,5))
    sns.scatterplot(x="network x", y="network y", hue=df.y.tolist(), s=8, ax=axes[0],
                    palette=col_map, legend=False,
                    data=df).set(title="NetworkSSC",xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    sns.scatterplot(x="ssc x", y="ssc y", hue=df.y.tolist(), s=8, ax=axes[1],
                    palette=col_map, legend=False,
                    data=df).set(title="SSC",xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    sns.scatterplot(x="adaptive x", y="adaptive y", hue=df.y.tolist(), s=8, ax=axes[2],
                    palette=col_map,
                    data=df).set(title="AdaptiveSSC",xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    axes[0].tick_params(bottom=False, left=False)
    axes[0].set_title("NetworkSSC", fontdict={'size': 18})
    axes[1].tick_params(bottom=False, left=False)
    axes[1].set_title("SSC", fontdict={'size': 18})
    axes[2].tick_params(bottom=False, left=False)
    axes[2].set_title("AdaptiveSSC", fontdict={'size': 18})

    fig.subplots_adjust(right=0.85)
    plt.legend(loc='right', bbox_to_anchor=(1.65, 0.5), ncol=1, title='Cell type',fontsize='15', title_fontsize='18',markerscale=2)
    os.makedirs("model_result/tsne_plots", exist_ok=True)
    plt.savefig("model_result/tsne_plots/{name}.jpg".format(name = dataname), dpi = 1000)
