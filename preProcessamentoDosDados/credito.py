import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

baseCredit = pd.read_csv('./credit_data.csv')

# retorna a quantidade de valores em default
print(np.unique(baseCredit['default'], return_counts=True))
# gera um gráfico com os dados da propriedade 'default' (precisa gerar o arquivo)
print(sns.countplot(x=baseCredit['default']))
# Faz um historiograma das idades que existem na base (precisa gerar o arquivo)
print(plt.hist(x=baseCredit["age"]))
# Faz um historiograma dos salários que existem na base (precisa gerar o arquivo)
print(plt.hist(x=baseCredit["income"]))
# Faz um historiograma das dividas que existem na base (precisa gerar o arquivo)
print(plt.hist(x=baseCredit["loan"]))

# gráfico idade vs salário
graphic = px.scatter_matrix(baseCredit, dimensions=['age', "income"])
graphic.show()

# Procura na base de dados todos os clientes com dade menor que 0
print(baseCredit[baseCredit['age'] < 0])

# Apaga a coluna 'age' inteira
# baseCredit2 = baseCredit.drop('age', axis=1)
# print(baseCredit2)

# Apaga apenas os registros que tem a idade negativa
baseCredit2 = baseCredit.drop(baseCredit[baseCredit['age'] < 0].index)
print(baseCredit2)

# pega a média das idades não negativas
# media = baseCredit['age'][baseCredit['age'] > 0].mean()

# # todos os registros com idades negativas são substituidos com a média das idades
# baseCredit[baseCredit['age'] < 0] = media

# soma a quantidade de vezes que um valor null aparece na tabela
print(baseCredit.isnull().sum())

# pega todos as colunas de idade que estão nulas e substituir com a média
baseCredit['age'].fillna(baseCredit['age'].mean(), inplace=True)
print('-------------------------------------------------')
print(baseCredit.loc[(baseCredit['clientid'] == 29) | (
    baseCredit['clientid'] == 31) | (baseCredit['clientid'] == 32)])
print('-------------------------------------------------')
# divide os previsores no eixo x
xCredit = baseCredit.iloc[:, 1:4].values
# divide as classes
yCredit = baseCredit.iloc[:, 4].values

print(xCredit[:, 0].min(), xCredit[:, 1].min(), xCredit[:, 2].min())
print(xCredit[:, 0].max(), xCredit[:, 1].max(), xCredit[:, 2].max())
# escalona os valores de X
scalerCredit = StandardScaler()
xCredit = scalerCredit.fit_transform(xCredit)
print(xCredit[:, 0].min(), xCredit[:, 1].min(), xCredit[:, 2].min())
print(xCredit[:, 0].max(), xCredit[:, 1].max(), xCredit[:, 2].max())

# Separa uma quantidade de registros para treinamento e outro para test
xCreditTreinamento, xCreditTest, yCreditTreinamento, yCreditTest = train_test_split(
    xCredit, yCredit, test_size=0.25, random_state=0)

# salva o arquivo modiciado
with open('credit.pkl', mode='wb') as f:
    pickle.dump([xCreditTreinamento, yCreditTreinamento,
                xCreditTest, yCreditTest], f)
