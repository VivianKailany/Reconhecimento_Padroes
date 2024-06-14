def acuracia(verdadeiros, previstos):
    predicoes_corretas = 0
    total_predicoes = len(verdadeiros)
    for verdadeiro, previsto in zip(verdadeiros, previstos):
        if verdadeiro == previsto:
            predicoes_corretas += 1
    return predicoes_corretas / total_predicoes

def precisao(verdadeiros, previstos, classe_positiva):
    verdadeiro_positivo = 0
    falso_positivo = 0
    for verdadeiro, previsto in zip(verdadeiros, previstos):
        if previsto == classe_positiva:
            if verdadeiro == classe_positiva:
                verdadeiro_positivo += 1
            else:
                falso_positivo += 1
    return verdadeiro_positivo / (verdadeiro_positivo + falso_positivo) if (verdadeiro_positivo + falso_positivo) != 0 else 0

def revocacao(rotulos_verdadeiros, previstos, classe_positiva):
    verdadeiro_positivo = 0
    falso_negativo = 0
    for verdadeiro, previsto in zip(rotulos_verdadeiros, previstos):
        if verdadeiro == classe_positiva:
            if previsto == classe_positiva:
                verdadeiro_positivo += 1
            else:
                falso_negativo += 1
    return verdadeiro_positivo / (verdadeiro_positivo + falso_negativo) if (verdadeiro_positivo + falso_negativo) != 0 else 0

def f1(precisao, revocacao):
    return 2 * (precisao * revocacao) / (precisao + revocacao) if (precisao + revocacao) != 0 else 0

def matriz_confusao(verdadeiros, previstos, num_classes):
    matriz = [[0] * num_classes for _ in range(num_classes)]
    for verdadeiro, previsto in zip(verdadeiros, previstos):
        verdadeiro = int(verdadeiro)
        previsto = int(previsto)
        matriz[verdadeiro][previsto] += 1
    return matriz
