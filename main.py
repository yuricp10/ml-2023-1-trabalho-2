import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


#funções para encontrar e remover os outliers
def outliers(dataFrame, nome):
    q1 = dataFrame[nome].quantile(0.25)
    q3 = dataFrame[nome].quantile(0.75)
    iqr = q3 - q1

    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    
    lista = dataFrame.index[ (dataFrame[nome] < lb) | (dataFrame[nome] > ub ) ]

    return lista



def remover_outliers(dataFrame, lista):

    lista = sorted(set(lista))
    dataFrame = dataFrame.drop(lista)
    
    return dataFrame


    


dados = pd.read_excel('dataset.xlsx')

# verificando dataset
print("verificando dataset:")
print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))
print(dados.head())

# analisando dados faltantes
print("analisando dados faltantes:")
print(dados.isnull().sum())


#como o dataset tem muitos valores vazios
#serão removidos aqueles individuos que não possuem todas as informações sobre o exame de sangue completas
# dados_limpo = dados.dropna(subset=['Hematocrit','Hemoglobin', 'Platelets','Mean platelet volume ', 'Red blood Cells','Lymphocytes','Mean corpuscular hemoglobin concentration (MCHC)','Leukocytes', 'Basophils','Mean corpuscular hemoglobin (MCH)','Eosinophils','Mean corpuscular volume (MCV)', 'Monocytes','Red blood cell distribution width (RDW)','Serum Glucose'])




print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))

# Removendo as colunas com mais de 90% de valores faltantes e o id do paciente
dados = dados.loc[:, dados.isnull().mean() < 0.8]

dados = dados.drop("Patient ID", axis=1)

print("O dataset possui {} linhas e {} colunas".format(dados.shape[0], dados.shape[1]))


print(dados.isnull().sum())

#usando IRQ para remover outliers
lista_index = []
for nome in dados.columns:
    print(nome)
    if dados[nome].dtypes == float:
        lista_index.extend(outliers(dados, nome))
        
    



df_final = remover_outliers(dados, lista_index)



#alterando os valores para que o dataset fique inteiramente numérico

for coluna in df_final.columns:
    if df_final[coluna].dtype == "object":
        df_final[coluna] = pd.Categorical(df_final[coluna]).codes



#corr_final = df_final.corr(numeric_only=True)

#Preenchendo lacunas vazias
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_final = pd.DataFrame(imputer.fit_transform(df_final), columns=df_final.columns)

#armazenando a correlação em um arquivo exel para algum tipo de análise visual 
#corr_final.to_excel(excel_writer='C:/Users/Lenovo/Desktop/testeExcel/arquivo.xlsx')
#remover a linha a cima caso queira testar com um diretório na propria maquina

#testando matshow para visualizar, porém sem muito sucesso. Seria necessário configurar melhor para uma melhor análise ou analisar atrávez do excel 

# plt.matshow(corr_final)
# plt.show()

# Separa os dados em conjunto de treinamento e teste

print(df_final)

X = df_final.drop("SARS-Cov-2 exam result", axis=1)
#X = pd.get_dummies(X)
y = df_final["SARS-Cov-2 exam result"]

# melhor_resultado = 0
# melhor_semente = 0
# mc = []



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#TESTANDO KNN
# Criação e treinamento do modelo KNN
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(X_test)
precisaoknn = accuracy_score(y_test, y_pred)
mcknn = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)


print('Acurácia do knn: {:.2f}%'.format(precisaoknn * 100))
print('Matriz de confusão:\n',mcknn)
print('Precisão:', precision)
print('Revocação:', recall)


#TESTANDO ÁRVORE DE DECISÃO
tree = DecisionTreeClassifier(max_depth=3)

# Treina o modelo usando o conjunto de treinamento
tree.fit(X_train, y_train)

# Faz a predição usando o conjunto de teste
y_pred = tree.predict(X_test)


# Avaliando modelo
precisaoAD = tree.score(X_test, y_test)
mcad = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)

print('Acurácia da árvore de decisão: {:.2f}%'.format(precisaoAD * 100))
print('Matriz de confusão: \n',mcad)
print('Precisão:', precision)
print('Revocação:', recall)


#TESTANDO NAIVE BAYES
# Instancia o modelo de Naive Bayes
modelo = GaussianNB()

# Treina o modelo
modelo.fit(X_train, y_train)

# Avalia a acurácia do modelo
y_pred = modelo.predict(X_test)
precisaoNB = accuracy_score(y_test, y_pred)
mcnb = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)

print('Acurácia do naive bayes: {:.2f}%'.format(precisaoNB * 100))
print('Matriz de confusão: \n',mcnb)
print('Precisão:', precision)
print('Revocação:', recall)