#%%
# Este projeto tem como objetivo desenvolver uma GCN - Graph Convolutional Networks, para classificar os nós de um grafo. 
# O dataset utilizado é o Cora, um conjunto de dados de citações científicas, 
# onde cada nó representa um artigo e as arestas representam as citações entre eles. 
# O objetivo é classificar os artigos em diferentes categorias com base em suas características e conexões.

#%%
# Importação de bibliotecas:
from torch_geometric.datasets import Planetoid # Importação de datasets do PyTorch Geometric
from torch_geometric.nn import GCNConv # Importação da camada de convolução para grafos
import torch.nn.functional as F # Importação de funções de ativação e perda
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
# Antes vamos pensar:
# 1. Qual é a tarefa?
# - Classificar nós?        → a saída é um label por nó
# - Classificar grafos?     → a saída é um label por grafo
# Prever arestas?         → a saída é um label por aresta
# 2. O que entra?
# - Features dos nós?       → sim → data.x
# - Features das arestas?   → tem ou não tem?
# - Estrutura do grafo?     → sempre → edge_index
# - Vários grafos?          → precisa de batch
# 3. O que sai?
# - Classificação?   → última camada tem num_classes neurônios
# - Regressão?       → última camada tem 1 neurônio

#%%

#%%
class GCN(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(GCN,self).__init__()
        self.conv1= GCNConv(in_channels,hidden_channels)
        self.conv2= GCNConv(hidden_channels,out_channels)

    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,p=0.5)
        x=self.conv2(x,edge_index)
        return x
#%% [markdown]
# Fluxo
# entrada:  [2708, 1433]
# conv1:    [2708, 16]    ← agregou vizinhos + aplicou pesos
# relu:     [2708, 16]    ← não-linearidade
# dropout:  [2708, 16]    ← regularização
# conv2:    [2708, 7]     ← uma pontuação por classe pra cada nó

#%%
def train(model, train_mask, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out=model(data.x,data.edge_index)
    loss=criterion(out[train_mask],data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss

#%%
criterion = torch.nn.CrossEntropyLoss()
model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#%%
data=dataset[0].to(device)
for epoch in range(200):
    loss = train(model,data.train_mask,data, optimizer, criterion)
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

#%%
def evaluate(model, data, mask):
    model.eval()                            # desativa dropout
    with torch.no_grad():                   # não calcula gradiente
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)            # classe com maior pontuação
        correct = pred[mask] == data.y[mask]
        return correct.sum() / mask.sum()   # acurácia

# %%
val_acc  = evaluate(model, data, data.val_mask)
test_acc = evaluate(model, data, data.test_mask)

print(f'Val Acc:  {val_acc:.4f}')
print(f'Test Acc: {test_acc:.4f}')
# %%
