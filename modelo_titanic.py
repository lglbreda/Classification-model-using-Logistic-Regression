# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:53:59 2026

@author: Luís Breda


MODELO DE CLASSIFICAÇÃO COM REGRESSÃO LOGÍSTICA

Este script aplica um modelo de regressão logística ao dataset Titanic,
incluindo o pré-processamento dos dados, divisão em treino e teste,
previsão e avaliação através da matriz de confusão e da taxa de acerto.
"""

# 1. Carregamento das bibliotecas
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# 2. Carregamento do dataset
passageiros_original = pd.read_csv("./dataset/titanic_v2.csv")

print("Passageiros com informação em branco:\n", passageiros_original.isnull().sum())

# 3. Pré-processamento: manter só variáveis relevantes
passageiros_original = passageiros_original[["survived", "age", "sex", "pclass"]]

# Remover apenas linhas com NA nestas colunas
passageiros_original.dropna(subset=["survived", "age", "sex", "pclass"], inplace=True)

print("Após tratamento, informação em branco:\n", passageiros_original.isnull().sum())

# Transformar variável categórica em numérica
passageiros_original["sex_num"] = passageiros_original["sex"].map({"male": 0, "female": 1})

# Selecionar variáveis do modelo
passageiros = passageiros_original[["survived", "age", "sex_num", "pclass"]]

# 4. Divisão treino/teste
passageiros_treino = passageiros.sample(frac=0.8, random_state=42)
passageiros_teste  = passageiros.drop(passageiros_treino.index)

x_teste = passageiros_teste[["age", "pclass", "sex_num"]]
y_teste = passageiros_teste["survived"]

# Treino do modelo (Statsmodels)
modelo_logistico_treino = smf.logit(
    formula="survived ~ age + pclass + sex_num",
    data=passageiros_treino
).fit()

print(modelo_logistico_treino.summary())

# 5. Previsão e avaliação
previsao_prob = modelo_logistico_treino.predict(x_teste)
pred_0_1 = [1 if p >= 0.5 else 0 for p in previsao_prob]

tabela_confusao = pd.crosstab(
    index=y_teste,
    columns=pd.Categorical(pred_0_1, categories=[0, 1]),
    rownames=["Observado"],
    colnames=["Previsto"]
)
print("\nMatriz de confusão:\n", tabela_confusao)

accuracy = (np.array(pred_0_1) == y_teste.values).mean()
print("\nTaxa global de acerto (Accuracy):", accuracy)

