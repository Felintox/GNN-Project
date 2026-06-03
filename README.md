# GCN - Classificação de Nós (Dataset Cora)
<img width="700" height="363" alt="gcn_web" src="https://github.com/user-attachments/assets/82f6223c-7a51-4795-b071-36698c040d9f" />

Este projeto implementa uma Rede Neural Convolucional em Grafos (Graph Convolutional Network - GCN) para classificar nós em um grafo. O modelo foi desenvolvido utilizando a biblioteca PyTorch Geometric.

## Objetivo

O objetivo principal é classificar artigos científicos em 7 diferentes categorias com base em suas características textuais e em suas conexões. 
No dataset, cada nó representa um artigo e as arestas representam as citações entre eles.

## Sobre o Dataset (Cora)

O dataset utilizado é o **Cora** (disponibilizado via `Planetoid` no PyTorch Geometric). Suas principais características são:
- **Nós (Artigos):** 2.708
- **Features por nó (Bag-of-words):** 1.433
- **Arestas (Citações):** 10.556
- **Classes:** 7

## Tecnologias Utilizadas

- Python
- PyTorch
- PyTorch Geometric
- Optuna

## Arquitetura do Modelo

A rede neural foi construída com duas camadas convolucionais para grafos (`GCNConv`):
1. **Camada 1:** Agrega os vizinhos e reduz a dimensionalidade das features (1433 -> `hidden_channels` variável).
2. **Ativação e Regularização:** Utiliza ativação não-linear `ReLU` e taxa de `Dropout` otimizada para evitar overfitting.
3. **Camada 2:** Gera a pontuação final para cada classe (`hidden_channels` -> 7 canais).

## Otimização de Hiperparâmetros

Este projeto utiliza o **Optuna** para buscar automaticamente os melhores hiperparâmetros. Durante a otimização (50 *trials*), são testados e ajustados:
- Taxa de aprendizado (`lr`)
- Número de canais ocultos (`hidden_channels`: 16, 32, 64 ou 128)
- Taxa de `Dropout`
- Decaimento de pesos (`weight_decay`)

Após encontrar a melhor combinação com base na acurácia de validação, o modelo final é treinado por 200 épocas utilizando o otimizador **Adam** e a função de custo **CrossEntropyLoss**, sendo posteriormente avaliado no conjunto de teste.

## Como executar

Certifique-se de ter as bibliotecas instaladas (PyTorch, PyTorch Geometric e Optuna) e execute o script Python:

```bash
pip install optuna
python cora-classification.py
```
