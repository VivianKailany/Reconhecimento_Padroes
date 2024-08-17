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


- **Máquinas de Vetores de Suporte (SVM)**
  
  O SVM é um classificador que busca encontrar o hiperplano que melhor separa as classes no espaço de características. Utiliza o conceito de margem máxima, que é a distância entre o hiperplano e os pontos de dados mais próximos de cada classe, chamados de vetores de suporte. O objetivo é maximizar essa margem.

  Para problemas não linearmente separáveis, o SVM pode usar o truque do kernel para mapear os dados para um espaço dimensional mais alto, onde uma separação linear pode ser encontrada. O kernel mais comum é o kernel radial (RBF), que é definido como:

  
  k(x_i, x_j) = exp(-gamma | x_i - x_j |^2)


  Onde \( gamma \) é um parâmetro que controla a largura do kernel.

  A função de decisão do SVM é dada por:

  
  f(x) = (b^* + sum_{i \in S^*} alpha_i y_i k(x_i, x))
  

  Onde \( S^* \) representa os vetores de suporte, \( alpha_i \) são os coeficientes Lagrange, e \( b^* \) é o termo de viés.


- **Árvore de Decisão**
  
  As árvores de decisão são modelos de aprendizado supervisionado que dividem o espaço de características em regiões distintas usando decisões baseadas em características. Cada nó da árvore representa uma decisão sobre uma característica, e cada ramo representa o resultado dessa decisão. As folhas da árvore representam as classes finais.

  O algoritmo de árvore de decisão funciona da seguinte forma:

  1. **Seleção da Melhor Característica**: Escolhe a característica que melhor separa os dados com base em critérios como a redução da impureza (por exemplo, índice de Gini ou entropia).
  
  2. **Divisão dos Dados**: Divide os dados em subconjuntos com base no valor da característica selecionada.
  
  3. **Recursão**: Aplica o mesmo processo recursivamente em cada subconjunto até que todas as características sejam usadas ou os dados sejam suficientemente homogêneos.

  A árvore de decisão resultante pode ser visualizada como um gráfico de fluxo, facilitando a interpretação das decisões tomadas pelo modelo.