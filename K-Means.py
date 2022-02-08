from sklearn.cluster import KMeans
import numpy as np, csv, matplotlib.pyplot as plt, pandas as pd

def transform(data):
    print("Transforming data...")
    x = []
    y = []
    for row in data:
        x.append(row[0])
        y.append(row[1])
    plt.scatter(x,y)
    plt.show()

def draw(data, labels):
    plt.xlim(-10, 300)
    plt.ylim(-80, 180)
    for i in range(len(labels)):
        point = data[i]
        if labels[i] == 0:
            plt.plot(point[0],point[1],'bo')
        elif labels[i] == 1:
            plt.plot(point[0],point[1],'r*')
        else:
            plt.plot(point[0],point[1],'yv')
    plt.show()
    
def k_means(data):
    X = np.array(data)
    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(X)
    draw(X, kmeans_model.labels_)

def read_file(file_name):
    data = list()
    with open(file_name, encoding='utf-8') as f:
        reader = csv.reader(f)
        for rows in reader:
            data.append(rows)
    #k_means(data)
    transform(np.array(data))

def main():
    read_file('PCA tfidf_Ch.csv')
    print("Finish!")

main()