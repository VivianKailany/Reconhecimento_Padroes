# Reconhecimento_Padroes

## Conteúdo

### Classificadores Estatísticos

- **Regressão Linear**
  
  A regressão linear é um método estatístico que modela a relação entre uma variável dependente \( y \) e uma variável independente \( X \) através de uma função linear:
  
     y = β0 + β1 ⋅X+ϵ
  
  Onde:\
  β0: é o intercepto da reta (valor de y quando X=0),\
  β1: é o coeficiente de inclinação da reta,\
  ϵ é o erro aleatório.

  Os coeficientes β0 e β1 são calculados usando métodos de mínimos quadrados, minimizando a soma dos quadrados dos resíduos (diferença entre os valores observados e os valores previstos).

- **Regressão Logística**
  
  A regressão logística é um método utilizado para modelar a probabilidade de uma variável dependente binária \( y \) em função de variáveis independentes \( X \) através da função logística:
  
  P(y=1∣X) = 1/(1+e−(β0+β1⋅X)
​

  
  Onde:\
  β0: é o intercepto da função logística,\
  β1: é o coeficiente associado à variável independente
  X,\
  e é a base do logaritmo natural.

- **Regressão Polinomial**
  
  A regressão polinomial é uma extensão da regressão linear, onde a relação entre a variável dependente \( y \) e a variável independente \( X \) é modelada como um polinômio de grau \( n \):
  
    y = β0+β1⋅X+β2⋅X2+…+βn⋅Xn+ϵ
  
  Onde:\
  β0, β1,…, βn são os coeficientes a serem estimados,\
  ϵ é o erro aleatório.

- **Classificadores Estatísticos**
  
  Os classificadores estatísticos são métodos que categorizam observações com base em informações estatísticas das variáveis independentes. Alguns exemplos incluem:
  
  Naive Bayes: Calcula a probabilidade de pertencer a uma classe dada a evidência fornecida pelas variáveis independentes, assumindo independência condicional entre as variáveis.
  
  Árvores de Decisão: Estruturam decisões em forma de árvore, dividindo o espaço de características em regiões distintas com base em regras estatísticas.
  
  K-Nearest Neighbors (KNN): Classifica novos pontos com base na maioria dos k vizinhos mais próximos no espaço de características, usando medidas de distância.
  
  Cada classificador possui métodos distintos de cálculo e ajuste, geralmente envolvendo técnicas estatísticas para decisão de classe.
