#%%
# Este projeto tem como objetivo desenvolver uma GCN - Graph Convolutional Networks, para classificar os nós de um grafo. 
# O dataset utilizado é o Cora, um conjunto de dados de citações científicas, 
# onde cada nó representa um artigo e as arestas representam as citações entre eles. 
# O objetivo é classificar os artigos em diferentes categorias com base em suas características e conexões.

#%%
# Importação de bibliotecas:
from torch_geometric.datasets import Planetoid # Importação de datasets do PyTorch Geometric
import torch.nn as nn 
import torch


#%%
# Confirmação da GPU

print(torch.cuda.is_available()) 

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%%
# Importação do dataset Cora:
dataset = Planetoid(root='~/somewhere/Cora', name='Cora')

# %% [markdown]
# Bag-of-words é uma forma de representar um texto como um vetor de números, onde você ignora a ordem das palavras e só registra a presença (ou frequência) delas.
# Isso gera um vetor de características para cada nó (artigo), onde cada posição do vetor representa a presença ou ausência de uma palavra específica no artigo.
# Neste exemplo como informado no enunciado:

# - 2708 Nós (Artigos)
# - 1433 Features em cada nó (Bag-of-words)
# - edge_index: 10556 arestas (Citações entre os artigos)
# %%
print(f'Número de Grafos: {len(dataset)}')
print(f'Número de Nós: {dataset[0].num_nodes}')
print(f'Número de Features: {dataset.num_node_features}')
print(f'Número de Classes: {dataset.num_classes}')
print(f'Número de Arestas: {dataset[0].num_edges}')

# %%
print(f'Nós de treinamento: {dataset[0].train_mask.sum()}')
print(f'Nós de validação: {dataset[0].val_mask.sum()}')
print(f'Nós de teste: {dataset[0].test_mask.sum()}')
# %% [markdown]
# Bom, o falo de ter poucos dados de treinamento é algo esperado devido a natureza do dataset:
# O Cora foi criado pra testar exatamente isso: "quanto a estrutura do grafo ajuda quando você tem pouquíssimos labels?"

#%%
class GCN(nn.Module):
