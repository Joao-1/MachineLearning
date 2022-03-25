import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

baseCensus = pd.read_csv('./census.csv')

# Pega todos as linhas e todas as colunas de 0 a 13 (os previsores)
xCensus = baseCensus.iloc[:, 0:14].values
# Pega todas as linhas e apenas a linha 14 (a classe)
yCensus = baseCensus.iloc[:, 14].values

# instância a classe LabelEncoder
labelEncoderWorkclass = LabelEncoder()
labelEncoderEducation = LabelEncoder()
labelEncoderMarital = LabelEncoder()
labelEncoderOccupation = LabelEncoder()
labelEncoderRelationship = LabelEncoder()
labelEncoderRace = LabelEncoder()
labelEncoderSex = LabelEncoder()
labelEncoderCountry = LabelEncoder()

# Passa as respectivas colunas para valores númericos
xCensus[:, 1] = labelEncoderWorkclass.fit_transform(xCensus[:, 1])
xCensus[:, 3] = labelEncoderEducation.fit_transform(xCensus[:, 3])
xCensus[:, 5] = labelEncoderMarital.fit_transform(xCensus[:, 5])
xCensus[:, 6] = labelEncoderOccupation.fit_transform(xCensus[:, 6])
xCensus[:, 7] = labelEncoderRelationship.fit_transform(xCensus[:, 7])
xCensus[:, 8] = labelEncoderRace.fit_transform(xCensus[:, 8])
xCensus[:, 9] = labelEncoderSex.fit_transform(xCensus[:, 9])
xCensus[:, 13] = labelEncoderCountry.fit_transform(xCensus[:, 13])

# Para evitar pesos errados, passa todos os valores númericos colocados anteriormente para valores binários de 0 ou 1
onehotencoderCensus = ColumnTransformer(transformers=[(
    'OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

xCensus = onehotencoderCensus.fit_transform(xCensus).toarray()

# Para também evitar pesos errados, passa todos os valores numericos anteriormente para a mesma escala usando padronização
scalerCensus = StandardScaler()
xCensus = scalerCensus.fit_transform(xCensus)

# Separa uma quantidade de registros para treinamento e outro para test
xCensusTreinamento, xCensusTest, yCensusTreinamento, yCensusTest = train_test_split(
    xCensus, yCensus, test_size=0.15, random_state=0)

# salva o arquivo modiciado
with open('census.pkl', mode='wb') as f:
    pickle.dump([xCensusTreinamento, yCensusTreinamento,
                xCensusTest, yCensusTest], f)
