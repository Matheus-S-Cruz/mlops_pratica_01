import pandas as pd


COLUNAS_OBRIGATORIAS = {
    "id_transacao",
    "id_cliente",
    "categoria",
    "valor",
}


def carregar_dados(caminho_csv: str = "data/raw_transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)

    colunas_faltantes = COLUNAS_OBRIGATORIAS - set(df.columns)
    if colunas_faltantes:
        raise ValueError(
            "O CSV não possui todas as colunas obrigatórias. "
            f"Colunas ausentes: {sorted(colunas_faltantes)}"
        )

    return df


def processar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas_faltantes = COLUNAS_OBRIGATORIAS - set(df.columns)
    if colunas_faltantes:
        raise ValueError(
            "O DataFrame não possui todas as colunas obrigatórias. "
            f"Colunas ausentes: {sorted(colunas_faltantes)}"
        )

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

    novas_colunas = []
    for metrica, categoria in pivot.columns:
        if metrica == "id_transacao":
            novas_colunas.append(f"frequencia_{categoria}")
        else:
            novas_colunas.append(f"gasto_{categoria}")

    pivot.columns = novas_colunas
    pivot = pivot.reset_index()

    colunas_frequencia = [
        coluna for coluna in pivot.columns
        if coluna.startswith("frequencia_")
    ]

    pivot["frequencia"] = pivot[colunas_frequencia].sum(axis=1)

    pivot["churn"] = (pivot["frequencia"] < 8).astype(int)

    return pivot


def carregar_e_processar(
    caminho_csv: str = "data/raw_transactions.csv",
) -> pd.DataFrame:
    df = carregar_dados(caminho_csv)
    return processar_dados(df)


if __name__ == "__main__":
    dados_processados = carregar_e_processar()
    print(dados_processados.head())
    print(f"\nClientes processados: {len(dados_processados)}")
    print("\nDistribuição de churn:")
    print(dados_processados["churn"].value_counts().sort_index())
