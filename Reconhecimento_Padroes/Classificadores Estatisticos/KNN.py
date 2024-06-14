import numpy as np

class KNN:
    def fit(self, X_treino, y_treino):
        self.X_treino = X_treino
        self.y_treino = y_treino
        
    def predict(self, X_test):
        previsoes = []
        for x in X_test:
            distancias = [np.linalg.norm(x - X_treino) for X_treino in self.X_treino]
            k_indices = np.argsort(distancias)[:3]  
            k_vizinhos = [self.y_treino[i] for i in k_indices]  
            previsao = max(set(k_vizinhos), key=k_vizinhos.count)  
            previsoes.append(previsao)
        return previsoes