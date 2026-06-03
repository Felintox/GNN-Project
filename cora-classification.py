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
import optuna


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
    def __init__(self,in_channels,hidden_channels,out_channels,dropout=0.5):
        super(GCN,self).__init__()
        self.conv1= GCNConv(in_channels,hidden_channels)
        self.conv2= GCNConv(hidden_channels,out_channels)
        self.dropout=dropout

    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
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
data=dataset[0].to(device)
#%%

# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], 
# train_mask=[2708], val_mask=[2708], test_mask=[2708])
#%%
def objective (trial):
    # definindo hiperparametros a serem otimizados
    lr= trial.suggest_float('lr',1e-4,1e-1,log=True)
    hidden_channels=trial.suggest_categorical('hidden_channels',[16,32,64,128   ])
    dropout=trial.suggest_float('dropout',0.2,0.7)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model= GCN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout
    ).to(device)

    optimizer=torch.optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay
                               )
    
    criterion=torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out=model(data.x,data.edge_index)
        loss=criterion(out[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:  # só a cada 50 épocas
            print(f"  Trial {trial.number} | Epoch {epoch+1} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
  
        out=model(data.x,data.edge_index)
        pred=out.argmax(dim=1)
        val_acc=(pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()
    
    return val_acc
    
#%%
study=optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=50,show_progress_bar=True)

bp=study.best_params

print(f"Melhores Hiperparâmetros: {bp}")
print(f"Melhor Acurácia de Validação: {study.best_value:.4f}")

#%% [markdown]
# Agora que temos os melhores hiperparâmetros, podemos treinar o modelo final e avaliar no conjunto de teste.

# %%
model= GCN(
    in_channels=data.num_node_features,
    hidden_channels=bp['hidden_channels'],
    out_channels=dataset.num_classes,
    dropout=bp['dropout']
).to(device)

optimizer=torch.optim.Adam(model.parameters(),
                           lr=bp['lr'],
                           weight_decay=bp['weight_decay']
                           )
criterion=torch.nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out=model(data.x,data.edge_index)
    loss=criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    out=model(data.x,data.edge_index)
    pred=out.argmax(dim=1)

test_acc=(pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
val_acc=(pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()

print(f"Test Acc: {test_acc:.4f}")
print(f"Val Acc: {val_acc:.4f}")
# %%
