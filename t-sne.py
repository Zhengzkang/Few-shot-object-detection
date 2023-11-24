import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import torch

# a = torch.load('a.pt')
# b = torch.load('b.pt')
# c = torch.load('c.pt')
# d = torch.load('d.pt')
# tensor = torch.cat((a,b,c,d), 0)
tensor = torch.load('tensor.pt')

x = torch.linspace(0, 0, 30)
y = torch.linspace(1, 1, 30)
z = torch.linspace(2, 2, 30)
m = torch.linspace(3, 3, 30)
n = torch.linspace(4, 4, 30)
k = torch.linspace(5, 5, 30)
b = torch.linspace(6, 6, 30)
labels = torch.cat((x,y,z,m,n,k,b), 0)

def plot_tsne(features, labels):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    plt.scatter(tsne_features[:30, 0], tsne_features[:30, 1],label='BulletTrain', marker='o')  # 将对降维的特征进行可视化
    plt.scatter(tsne_features[30:60, 0], tsne_features[30:60, 1],label='Pedestrain', marker='v')
    plt.scatter(tsne_features[60:90, 0], tsne_features[60:90, 1],label='RailwayStraight', marker='*')
    plt.scatter(tsne_features[90:120, 0], tsne_features[90:120, 1],label='RailwayLeft', marker='h')
    plt.scatter(tsne_features[120:150, 0], tsne_features[120:150, 1],label='RailwayRight',marker='d')
    plt.scatter(tsne_features[150:180, 0], tsne_features[150:180, 1],label='Helmet', marker='s')
    plt.scatter(tsne_features[180:210, 0], tsne_features[180:210, 1],label='Spanner', marker='p')
    # plt.scatter(tsne_features[:, 0], tsne_features[:, 1])
    plt.legend(fontsize=8)
    plt.xticks([])
    plt.yticks([])


    plt.show()






if __name__ == '__main__':
    features, labels = tensor, labels
    print(labels.shape)
    plot_tsne(features, labels)
