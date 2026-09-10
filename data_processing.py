import pandas as pd


COLUNAS_OBRIGATORIAS = {
    "id_transacao",
    "id_cliente",
    "categoria",
    "valor",
}


def carregar_dados(caminho_csv: str = "data/raw_transactions.csv") -> pd.DataFrame:
    """
    Lê o arquivo CSV bruto de transações.

    Parâmetros
    ----------
    caminho_csv : str
        Caminho do arquivo raw_transactions.csv.

    Retorno
    -------
    pd.DataFrame
        DataFrame contendo as transações brutas.
    """
    df = pd.read_csv(caminho_csv)

    colunas_faltantes = COLUNAS_OBRIGATORIAS - set(df.columns)
    if colunas_faltantes:
        raise ValueError(
            "O CSV não possui todas as colunas obrigatórias. "
            f"Colunas ausentes: {sorted(colunas_faltantes)}"
        )

    return df


def processar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma as transações em uma base por cliente.

    O processamento:
    1. Usa pd.pivot_table para calcular:
       - frequência de compras por categoria;
       - gasto total por categoria.
    2. Calcula a frequência total de compras de cada cliente.
    3. Gera a coluna churn:
       - 1 quando a frequência total for menor que 8;
       - 0 caso contrário.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de transações brutas.

    Retorno
    -------
    pd.DataFrame
        Base agregada por cliente, pronta para ser usada no treinamento.
    """
    colunas_faltantes = COLUNAS_OBRIGATORIAS - set(df.columns)
    if colunas_faltantes:
        raise ValueError(
            "O DataFrame não possui todas as colunas obrigatórias. "
            f"Colunas ausentes: {sorted(colunas_faltantes)}"
        )

    # Tabela dinâmica obrigatória do trabalho:
    # conta as transações e soma os gastos por categoria para cada cliente.
    pivot = pd.pivot_table(
        df,
        index="id_cliente",
        columns="categoria",
        values=["id_transacao", "valor"],
        aggfunc={
            "id_transacao": "count",
            "valor": "sum",
        },
        fill_value=0,
    )

    # Achata o MultiIndex criado pelo pivot_table.
    novas_colunas = []
    for metrica, categoria in pivot.columns:
        if metrica == "id_transacao":
            novas_colunas.append(f"frequencia_{categoria}")
        else:
            novas_colunas.append(f"gasto_{categoria}")

    pivot.columns = novas_colunas
    pivot = pivot.reset_index()

    # Frequência total de compras por cliente.
    colunas_frequencia = [
        coluna for coluna in pivot.columns
        if coluna.startswith("frequencia_")
    ]

    pivot["frequencia"] = pivot[colunas_frequencia].sum(axis=1)

    # Regra de churn definida no enunciado.
    pivot["churn"] = (pivot["frequencia"] < 8).astype(int)

    return pivot


def carregar_e_processar(
    caminho_csv: str = "data/raw_transactions.csv",
) -> pd.DataFrame:
    """
    Função principal do módulo, pensada para ser importada pelo main.py.
    """
    df = carregar_dados(caminho_csv)
    return processar_dados(df)


if __name__ == "__main__":
    dados_processados = carregar_e_processar()
    print(dados_processados.head())
    print(f"\nClientes processados: {len(dados_processados)}")
    print("\nDistribuição de churn:")
    print(dados_processados["churn"].value_counts().sort_index())
