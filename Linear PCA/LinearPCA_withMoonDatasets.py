import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def extractData(file):
    data = pd.read_csv(file)
    return data

def splitXY(df):
    x0 = []
    x1 = []
    x = df.iloc[:,0:2].values 
    y = df.iloc[:,2].values
    for i in range(0, len(df)):
        if (df.iloc[:,2].values[i] == 0.0):
            x0.append(df.iloc[:,0:2].values[i])
        elif (df.iloc[:,2].values[i] == 1.0):
            x1.append(df.iloc[:,0:2].values[i])
    return x,y,x0,x1

if __name__ == "__main__":
    df = extractData("Data.csv")
    x, y ,x0, x1 = splitXY(df)
    xMean = np.mean(x,axis =0)
    xNom = x - xMean
    xNom_Matrix = np.matrix(xNom)
    covariance = (xNom_Matrix.transpose()*xNom_Matrix)/(x.shape[0]-1)
    w, v = np.linalg.eig(covariance)
    pca =v[:,:2]
    x0_pca = np.asarray(x0*pca)
    x1_pca = np.asarray(x1*pca)
    plt.scatter(np.asarray(x0_pca[:,0]), np.asarray(x0_pca[:,1]))
    plt.scatter(np.asarray(x1_pca[:,0]), np.asarray(x1_pca[:,1]))
    plt.show()