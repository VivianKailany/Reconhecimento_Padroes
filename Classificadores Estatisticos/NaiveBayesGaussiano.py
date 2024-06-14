import numpy as np


class NaiveBayesGaussiano:
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classe_prior = {}
        self.media = {}
        self.matriz_covariancia = {}
        for c in self.classes:
            X_c = X[y == c]
            self.classe_prior[c] = 1 / len(self.classes)
            self.media[c] = self.calcular_media(X_c)
            self.matriz_covariancia[c] = self.calcular_covariancia(X_c)

    def calcular_media(self, X):
        media = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            media[i] = np.sum(X[:, i]) / X.shape[0]
        return media

    def calcular_covariancia(self, X):
        covariancia = np.zeros((X.shape[1], X.shape[1]))
        media = self.calcular_media(X)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                covariancia[i, j] = np.sum((X[:, i] - media[i]) * (X[:, j] - media[j])) / (X.shape[0] - 1)
        return covariancia
    

    def predict(self, X):
        previsoes = []
        for x in X:
            posteriores = []
            for c in self.classes:
                prior = np.log(self.classe_prior[c])
                media = self.media[c]
                cov = self.matriz_covariancia[c]
                exponencial = -0.5 * np.dot(np.dot((x - media).T, np.linalg.inv(cov)), x - media)
                posterior = prior + exponencial 
                posteriores.append(posterior)
                
            previsoes.append(self.classes[np.argmax(posteriores)])
              
        return previsoes