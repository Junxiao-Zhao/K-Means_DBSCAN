from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import csv, numpy as np, matplotlib.pyplot as plt

def dbscan(data):
    print("Clustering(DBSCAN)...")
    dbscan_model = DBSCAN(3)
    dbscan_model.fit(data)
    visualize(data, dbscan_model.labels_, "DBSCAN")

def visualize(data, labels, type):
    print("Drawing Clusters("+type+")...")
    plt.suptitle("Cluster("+type+")")
    plt.xticks(np.arange(-20, 300, 10))
    plt.yticks(np.arange(-80, 180, 10))
    for i in range(len(labels)):
        point = data[i]
        if labels[i] == -1:
            plt.plot(point[0],point[1],'k^')
        elif labels[i] == 0:
            plt.plot(point[0],point[1],'bo')
        elif labels[i] == 1:
            plt.plot(point[0],point[1],'r*')
        else:
            plt.plot(point[0],point[1],'yv')
    plt.savefig(type + ".png")
    plt.show()

def k_means(data):
    print("Clustering(K-Means)...")
    kmeans_model = KMeans(n_clusters=2)
    kmeans_model.fit(data)
    visualize(data, kmeans_model.labels_, "K-Means")

def transform(data):
    x = []
    y = []
    for row in data:
        x.append(row[0])
        y.append(row[1])
    return [x,y]

def draw(data):
    print("Drawing Scatter Plot...")
    plt.suptitle('PCA of tf*idf of THUCNews')
    plt.scatter(data[0], data[1])
    plt.savefig("PCA tfidf_Ch.png")
    plt.show()

def write_file(data):
    print("Writing files...")
    with open('PCA tfidf_Ch.csv','w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        for rows in data:
            writer.writerow(rows)

def P_C_A(data):
    print("PCAing...")
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca.fit_transform(data)

def read_file(file_name):
    print("Reading files...")
    data = list()
    with open(file_name, encoding='utf-8') as f:
        reader = list(csv.reader(f)) 
        length = len(reader[0])
        for i in range(length):
            data.append(list())
        for column in reader[1:]:
            for i in range(1, length):
                data[i].append(column[i])
    data = P_C_A(np.array(data[1:], dtype=object))
    write_file(data)
    draw(transform(data))
    k_means(data)
    dbscan(data)

def main():
    read_file(r'D:\Shaw\Documents\Miscellany\网课\R计划数据挖掘方向\课程\Homework 1\tfidf table_Ch.csv')
    print("Finish!")

main()