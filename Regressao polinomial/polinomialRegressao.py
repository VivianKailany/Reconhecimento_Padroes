import numpy as np

class RegressaoPolinomialSemRegularizacao():
    def __init__(self, grau, a, epocas):
        self.grau = grau
        self.a = a
        self.epocas = epocas
        self.w = None

    def fit_gd(self, X, y):
        N = X.shape[0]
        X_poly = self.gerar_features_polinomiais(X)
        self.w = np.zeros((X_poly.shape[1], 1))

        for _ in range(self.epocas):
            # Calcular as previsões
            y_pred = np.dot(X_poly, self.w)
            
            erro = y_pred - y.reshape(-1, 1)
            grad = np.dot(X_poly.T, erro) / N
            
            self.w -= self.a * grad

    def predict(self, X):
        X_poly = self.gerar_features_polinomiais(X)
        # Calculando as previsões
        y_pred = np.dot(X_poly, self.w)

        return y_pred.flatten()

    def gerar_features_polinomiais(self, X):
        N = X.shape[0]
        X_poly = np.ones((N, 1)) 

        for d in range(1, self.grau + 1):
            X_poly = np.hstack((X_poly, X ** d))

        return X_poly

    def calcular_rmse(self, y_real, y_pred):
        rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
        return rmse



class RegressaoPolinomialComRegularizacao():
    def __init__(self, grau, a, epocas, alpha):
        self.grau = grau
        self.a = a
        self.epocas = epocas
        self.alpha = alpha
        self.w = None

    def fit_gd(self, X, y):
        N = X.shape[0]
        X_poly = self.gerar_features_polinomiais(X)
        self.w = np.zeros((X_poly.shape[1], 1))

        for _ in range(self.epocas):
            
            y_pred = np.dot(X_poly, self.w)
            erro = y_pred - y.reshape(-1, 1)
            grad = np.dot(X_poly.T, erro) / N
            
            self.w = (1 - self.a * self.alpha) * self.w - self.a * grad

    def predict(self, X):
        X_poly = self.gerar_features_polinomiais(X)
        
        y_pred = np.dot(X_poly, self.w)
        return y_pred.flatten()

    def gerar_features_polinomiais(self, X):
        N = X.shape[0]
        X_poly = np.ones((N, 1))  

        for d in range(1, self.grau + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def calcular_rmse(self, y_real, y_pred):
        rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
        return rmse

