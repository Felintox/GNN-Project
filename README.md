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

## Arquitetura do Modelo

A rede neural foi construída com duas camadas convolucionais para grafos (`GCNConv`):
1. **Camada 1:** Agrega os vizinhos e reduz a dimensionalidade das features (1433 -> 16 canais).
2. **Ativação e Regularização:** Utiliza ativação não-linear `ReLU` e `Dropout` (p=0.5) para evitar overfitting.
3. **Camada 2:** Gera a pontuação final para cada classe (16 -> 7 canais).

O modelo é treinado por 200 épocas utilizando o otimizador **Adam** e a função de custo **CrossEntropyLoss**.

## Como executar

Certifique-se de ter as bibliotecas instaladas (PyTorch e PyTorch Geometric) e execute o script Python:

```bash
python cora-classification.py
```
