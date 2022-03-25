import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

with open('./baseCredit/credit.pkl', 'rb') as f:
    xCreditTreinamento, yCreditTreinamento, xCreditTest, yCreditTest = pickle.load(
        f)

naiveCreditData = GaussianNB()
naiveCreditData.fit(xCreditTreinamento, yCreditTreinamento)

previsoes = naiveCreditData.predict(xCreditTest)

print(accuracy_score(yCreditTest, previsoes))

cm = ConfusionMatrix(naiveCreditData)
cm.fit(xCreditTreinamento, yCreditTreinamento)
