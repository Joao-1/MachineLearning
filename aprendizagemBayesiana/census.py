import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

with open('./census/census.pkl', 'rb') as f:
    xCensusTreinamento, yCensusTreinamento, xCensusTest, yCensusTest = pickle.load(
        f)

naiveCreditData = GaussianNB()
naiveCreditData.fit(xCensusTreinamento, yCensusTreinamento)

previsoes = naiveCreditData.predict(xCensusTest)

print(accuracy_score(yCensusTest, previsoes))

cm = ConfusionMatrix(naiveCreditData)
cm.fit(xCensusTreinamento, yCensusTreinamento)
