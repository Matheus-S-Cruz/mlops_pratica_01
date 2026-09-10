import pandas as pd

from data_processing import carregar_e_processar
from model_training import treinar_modelo


def main():
    print("=== PIPELINE DE CLASSIFICAÇÃO DE CHURN ===\n")

    print("1. Carregando e processando os dados...")

    dados = carregar_e_processar()

    print(f"Clientes processados: {len(dados)}")

    print("\n2. Treinando o modelo...")

    modelo, metricas = treinar_modelo(dados)

    print(f"\nAcurácia: {metricas['acuracia']:.2%}")

    print("\nMatriz de Confusão:")
    print(metricas["matriz_confusao"])

    print("\nRelatório de Classificação:")
    print(metricas["relatorio"])

    print("\n3. Realizando previsão para um novo cliente...")

    features = metricas["features"]

    novo_cliente = pd.DataFrame(
        [{
            "frequencia_Alimentos": 3,
            "frequencia_Casa": 2,
            "frequencia_Eletronicos": 2,
            "frequencia_Livros": 1,
            "frequencia_Roupas": 2,

            "gasto_Alimentos": 150.00,
            "gasto_Casa": 300.00,
            "gasto_Eletronicos": 1200.00,
            "gasto_Livros": 100.00,
            "gasto_Roupas": 250.00
        }]
    )

    novo_cliente = novo_cliente[features]

    previsao = modelo.predict(novo_cliente)

    if previsao[0] == 1:
        print("\nResultado: CLIENTE COM RISCO DE CHURN")
    else:
        print("\nResultado: CLIENTE SEM RISCO DE CHURN")


if __name__ == "__main__":
    main()