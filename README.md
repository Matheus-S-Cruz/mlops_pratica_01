````markdown
# Pipeline Modularizado de Classificação de Churn

Projeto desenvolvido para a atividade prática da Unidade Curricular de **Engenharia de Dados e MLOps**.

O objetivo do projeto é construir um pipeline modularizado em Python capaz de processar dados de transações de clientes, gerar informações agregadas, treinar um modelo de classificação e realizar uma previsão de churn para um novo cliente.

## Equipe

- **Pessoa A - Engenharia de Dados:** Matheus Silva da Cruz
- **Pessoa B - Ciência de Dados:** Nicolas André Krause de Mello
- **Pessoa C - MLOps e Integração:** João Paulo Araujo Cappeletti

## Estrutura do projeto

```text
mlops_pratica_01/
│
├── data/
│   └── raw_transactions.csv
│
├── data_processing.py
├── model_training.py
├── main.py
└── README.md
````

## Arquivos

### `data_processing.py`

Responsável pelo tratamento e transformação dos dados.

O módulo:

* lê o arquivo `raw_transactions.csv`;
* utiliza `pd.pivot_table`;
* calcula a frequência de compras por categoria;
* calcula o total gasto por categoria;
* calcula a frequência total de compras de cada cliente;
* cria a variável `churn`.

A regra utilizada para definir churn é:

```text
frequência < 8  -> churn = 1
frequência >= 8 -> churn = 0
```

---

### `model_training.py`

Responsável pelo treinamento e avaliação do modelo de classificação.

O módulo:

* recebe os dados processados;
* separa as variáveis de entrada e a variável alvo;
* divide os dados em 70% para treinamento e 30% para teste;
* utiliza um modelo `RandomForestClassifier`;
* realiza as previsões;
* calcula métricas de avaliação.

As principais métricas exibidas são:

* acurácia;
* matriz de confusão;
* relatório de classificação.

---

### `main.py`

É o arquivo principal do projeto.

Ele integra os módulos de processamento e treinamento e executa o pipeline completo.

O fluxo é:

```text
raw_transactions.csv
        ↓
data_processing.py
        ↓
dados processados
        ↓
model_training.py
        ↓
modelo treinado
        ↓
main.py
        ↓
previsão de churn
```

Ao final, o programa cria um exemplo de novo cliente e utiliza o modelo treinado para classificar se ele apresenta ou não risco de churn.

## Como executar

É necessário possuir o Python instalado.

Primeiro, instale as bibliotecas utilizadas:

```bash
pip install pandas scikit-learn
```

Depois, abra o terminal na pasta principal do projeto e execute:

```bash
python main.py
```

O `main.py` executa automaticamente o fluxo completo do pipeline.

## Execuções individuais

Também é possível testar os módulos separadamente.

### Processamento dos dados

```bash
python data_processing.py
```

### Treinamento do modelo

```bash
python model_training.py
```

### Pipeline completo

```bash
python main.py
```

## Tecnologias utilizadas

* Python
* Pandas
* Scikit-learn
* Random Forest
* VS Code
* Git
* GitHub

## Objetivo da solução

A solução busca identificar clientes com risco de churn utilizando informações históricas de frequência de compras e gastos por categoria.

O projeto também demonstra a separação de responsabilidades em um pipeline modularizado, dividindo as etapas entre Engenharia de Dados, Ciência de Dados e integração do modelo.

```
```
