import numpy as np
from metricas import acuracia, precisao, revocacao, f1

def validacao_cruzada(modelo, X, y, num_folds=10):
    fold_size = len(X) // num_folds
    acuracias = []
    precisoes = []
    revocacoes = []
    f1_scores = []

    for i in range(num_folds):
        inicio_fold = i * fold_size
        fim_fold = (i + 1) * fold_size

        X_validacao = X[inicio_fold:fim_fold]
        y_validacao = y[inicio_fold:fim_fold]

        X_treinamento = np.concatenate([X[:inicio_fold], X[fim_fold:]])
        y_treinamento = np.concatenate([y[:inicio_fold], y[fim_fold:]])

        modelo.fit_gd(X_treinamento, y_treinamento)
        previsoes = modelo.predict(X_validacao)
        
        acuracia_fold = acuracia(y_validacao, previsoes)
        acuracias.append(acuracia_fold)
        
        precisao_fold = precisao(y_validacao, previsoes, classe_positiva=1)
        precisoes.append(precisao_fold)
        
        revocacao_fold = revocacao(y_validacao, previsoes, classe_positiva=1)
        revocacoes.append(revocacao_fold)
        
        f1_fold = f1(precisao_fold, revocacao_fold)
        f1_scores.append(f1_fold)

    return acuracias, precisoes, revocacoes, f1_scores

def calcular_estatisticas_metricas(metricas):
    media_metricas = np.mean(metricas)
    desvio_padrao_metricas = np.std(metricas)
    return media_metricas, desvio_padrao_metricas
