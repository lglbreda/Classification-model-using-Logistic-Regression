#  LB_Caso_Pratico_Modelo_Reg_Log

## Descrição do Projeto

Este projeto implementa um **modelo de classificação utilizando
Regressão Logística** aplicado ao dataset **titanic_v2.csv**.

O objetivo é prever a sobrevivência dos passageiros com base nas
seguintes variáveis explicativas:

-   Idade (`age`)
-   Sexo (`sex`)
-   Classe do bilhete (`pclass`)

O modelo é desenvolvido em Python utilizando a biblioteca
**Statsmodels**.

------------------------------------------------------------------------

## Metodologia

### 1️-Carregamento das Bibliotecas

-   numpy
-   pandas
-   statsmodels

### 2️-Carregamento do Dataset

O dataset utilizado:

    ./dataset/titanic_v2.csv

O script verifica inicialmente a existência de valores em falta.

------------------------------------------------------------------------

### 3️-Pré-processamento dos Dados

-   Remoção de valores nulos (`dropna`)
-   Conversão da variável categórica `sex` em variável numérica:
    -   male → 0\
    -   female → 1
-   Seleção das variáveis relevantes para o modelo:
    -   survived
    -   age
    -   sex_num
    -   pclass

------------------------------------------------------------------------

### 4️-Divisão Treino/Teste

-   80% dos dados para treino
-   20% dos dados para teste
-   `random_state=42` para garantir reprodutibilidade

------------------------------------------------------------------------

### 5️-Treino do Modelo

Foi aplicada uma regressão logística com a seguinte fórmula:

    survived ~ age + pclass + sex_num

Utilizando:

    statsmodels.formula.api.logit()

------------------------------------------------------------------------

### 6️-Avaliação do Modelo

O desempenho do modelo é avaliado através de:

-   Matriz de confusão
-   Taxa global de acerto (Accuracy)

------------------------------------------------------------------------

## Estrutura do Projeto

    LB_Caso_Pratico_Modelo_Reg_Log/
    │
    ├── dataset/
    │   └── titanic_v2.csv
    │
    ├── modelo_titanic.py
    │
    └── README.md

------------------------------------------------------------------------

## Como Executar

1.  Garantir que o dataset está dentro da pasta `dataset`
2.  Instalar as dependências necessárias:

```{=html}
<!-- -->
```
    pip install numpy pandas statsmodels

3.  Executar o script:

```{=html}
<!-- -->
```
    python modelo_titanic.py

------------------------------------------------------------------------

## Resultados Esperados

O script apresenta:

-   Resumo estatístico do modelo logístico
-   Matriz de confusão
-   Taxa global de acerto (Accuracy)

------------------------------------------------------------------------

## Análise de Dados em Falta

Na análise inicial do dataset foram identificados valores em branco nas
seguintes variáveis:

-   `age`: 177 valores em falta\
-   `cabin`: 687 valores em falta\ não foi considerado para análise
-   `embarked`: 2 valores em falta

Para o modelo foram consideradas apenas as variáveis relevantes
(`survived`, `age`, `sex`, `pclass`), tendo sido feito o tratamento dos
valores em falta.

Após o tratamento, o conjunto final utilizado no modelo não apresentou
quaisquer valores em branco.

------------------------------------------------------------------------

## Modelo Utilizado

Foi aplicado um modelo de **Regressão Logística (Logit)** para prever a
variável `survived`.

-   Nº de observações: **571**
-   Variáveis explicativas: `age`, `pclass`, `sex`
-   O modelo convergiu com sucesso

------------------------------------------------------------------------

## Interpretação dos Coeficientes

Todos os coeficientes apresentaram **p-value \< 0.001**, indicando
elevada significância estatística.

### Idade (`age`)

-   Coeficiente negativo (-0.0410). Se o coeficiente é negativo 
    significa que quando essa variável aumenta, a probbilidade de 
    sobreviver diminui.
-   Passageiros mais velhos tiveram menor probabilidade de
    sobrevivência.
-   Cada aumento de 1 ano na idade reduz a probabilidade de
    sobrevivência (mantendo as restantes variáveis constantes).

### Classe (`pclass`)

-   Coeficiente negativo (-1.3436). Se o coeficiente é negativo, significa
    que quando a classe sobe a probabilidade de sobrevivência diminui.
-   Passageiros de classes mais baixas, neste caso de 2ª e 3ª classe,  
    tiveram menor probabilidade de sobreviver.
-   A classe social teve forte impacto na sobrevivência, ainda mais que idade
    porque o valor é ainda mais pequeno.

### Sexo (`sex_num`)

-   Coeficiente positivo (2.5812). Quanto maior o valor da variável, maior
    a probabilidade de sobreviver.
-   Como foi codificado (0 ->1) ou seja (homem ->mulher), então quando passa de 0 para 1
    a probabilidade de sobrevivência aumenta muito.    
-   Mulheres tiveram probabilidade significativamente maior de
    sobreviver.
-   Esta variável apresenta o maior impacto no modelo. A probabilidade de sobreviver
    aumenta muito se for mulher.
-   Quanto maior o valor absoluto do coeficiente, maior o impacto no modelo.

### Em resumo

O impacto pode ser resumido da seguinte forma
-   (`age`)     ->  pequeno
-   (`pclass`)  ->  forte
-   (`sex_num`) ->  muito forte

Uma forma intuitiva de pensar, para ajudar a perceber os coeficientes em relação
a valores negativos, positivos ou absolutos pode ser o seguinte:

Se estivermos a subir uma montanha (probabilidade de sobreviver):
-   Sexo feminino -> dá um empurrão gigante para cima
-   3ª classe -> puxa bastante para baixo
-   Ser mais velho -> puxa um bocadinho para baixo

------------------------------------------------------------------------

## Desempenho do Modelo

### Matriz de Confusão

                Previsto 0   Previsto 1
  ------------- ------------ ------------
  Observado 0   72           11
  Observado 1   20           40

### Accuracy (Taxa Global de Acerto)

----->   78,32%   <-----

O modelo apresenta um bom nível de desempenho para um modelo simples com
apenas três variáveis explicativas.

------------------------------------------------------------------------

## Conclusão Final

O modelo confirma a história amplamente documentada no desastre
do Titanic:

-   Mulheres tiveram maior probabilidade de sobrevivência.
-   Passageiros mais jovens tiveram vantagem na sobrevivência, 
    incluindo bébés e crianças.
-   Passageiros de classes superiores tiveram maior probabilidade de
    sobreviver.
-   "Mulheres e crianças primeiro".

Com apenas três variáveis, o modelo conseguiu atingir uma taxa de acerto
superior a 78%, demonstrando que fatores demográficos simples já
explicam uma parte significativa da sobrevivência

## Autor

Luís Breda\
Janeiro 2026