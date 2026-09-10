import pandas as pd
from typing import Tuple, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_processing import carregar_e_processar


def treinar_modelo(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Realiza a divisão treino/teste (70/30), treina o modelo de classificação
    e retorna o modelo treinado com suas métricas.
    """
    if target_col not in df.columns:
        raise ValueError(f"A coluna alvo '{target_col}' não está presente no DataFrame.")

    # Remove identificadores e variáveis que causam vazamento de dados (data leakage)
    colunas_ignorar = [target_col, "id_cliente", "frequencia"]
    features = [col for col in df.columns if col not in colunas_ignorar]

    X = df[features]
    y = df[target_col]

    # Divisão 70% Treino e 30% Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )

    # Modelo de Classificação
    modelo = RandomForestClassifier(n_estimators=100, random_state=random_state)
    modelo.fit(X_train, y_train)

    # Predição e Avaliação
    y_pred = modelo.predict(X_test)

    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "matriz_confusao": confusion_matrix(y_test, y_pred),
        "relatorio": classification_report(y_test, y_pred),
        "features": features
    }

    return modelo, metricas


# Bloco responsável por imprimir no terminal quando você roda 'python model_training.py'
if __name__ == "__main__":
    print("--- Treinando o Modelo (Pessoa B) ---")
    
    # 1. Carrega os dados tratados pela Pessoa A
    dados = carregar_e_processar()
    
    # 2. Treina o modelo e obtém os resultados
    modelo_treinado, resultados = treinar_modelo(dados)
    
    # 3. Exibe os resultados no terminal
    print(f"\nAcurácia do Modelo: {resultados['acuracia']:.2%}\n")
    print("Matriz de Confusão:")
    print(resultados["matriz_confusao"])
    print("\nRelatório de Classificação:")
    print(resultados["relatorio"])