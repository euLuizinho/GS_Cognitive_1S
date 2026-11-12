"""
src/evaluate.py
===============
Avalia o modelo YOLO treinado no conjunto de teste/validação e consolida métricas.

Por que existe:
    Para garantir rigor técnico e comprovação empírica na arguição da GS,
    nenhuma métrica pode ser inventada ou anotada manualmente. Este script
    executa a validação formal no split de teste, exporta results/metricas.json
    e copia/gera a curva Precisão-Recall e Matriz de Confusão para a pasta results/.

Uso:
    python src/evaluate.py --weights models/best.pt --data data/data.yaml --split test
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Funções com responsabilidade única
# ──────────────────────────────────────────────────────────────────────────────

def carregar_modelo_yolo(caminho_pesos: Path) -> YOLO:
    """
    Carrega o modelo YOLO a partir do arquivo de pesos especificado.

    Por que existe:
        Isola o carregamento do modelo e valida se o arquivo de pesos realmente
        existe antes de iniciar qualquer operação computacional custosa.

    Args:
        caminho_pesos: Path para o arquivo .pt (ex: models/best.pt).

    Returns:
        Instância do modelo YOLO carregada.

    Raises:
        FileNotFoundError: Se o arquivo de pesos não for encontrado.
    """
    if not caminho_pesos.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos não encontrado em: {caminho_pesos.resolve()}.\n"
            f"Execute o treino no Colab e baixe o best.pt para a pasta models/."
        )
    return YOLO(str(caminho_pesos))


def verificar_split_disponivel(caminho_yaml: Path, split_desejado: str) -> str:
    """
    Verifica se o split desejado existe no data.yaml; caso contrário, define fallback.

    Por que existe:
        Muitos datasets não têm split 'test' explícito ou nomeiam como 'val'/'valid'.
        Esta função evita erros em tempo de execução selecionando automaticamente
        o melhor split disponível para avaliação formal.

    Args:
        caminho_yaml: Path para o arquivo data.yaml.
        split_desejado: Nome do split pretendido ('test' ou 'val').

    Returns:
        Nome do split seguro a ser utilizado ('test' ou 'val').
    """
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        dados_yaml = yaml.safe_load(f)

    if split_desejado in dados_yaml and dados_yaml[split_desejado]:
        return split_desejado

    if "val" in dados_yaml and dados_yaml["val"]:
        print(f"[AVISO] Split '{split_desejado}' não encontrado no YAML. Usando 'val' como fallback.")
        return "val"

    raise ValueError(f"Nenhum split de validação/teste válido encontrado em {caminho_yaml}")


def executar_avaliacao(
    modelo: YOLO,
    caminho_yaml: Path,
    split: str = "test",
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
) -> object:
    """
    Executa a rotina de validação oficial do Ultralytics no dataset especificado.

    Por que existe:
        Executa a inferência em todo o conjunto de dados, aplicando cálculo
        padronizado de IoU e Non-Maximum Suppression (NMS) sob os limiares definidos.

    Args:
        modelo: Instância do modelo YOLO.
        caminho_yaml: Caminho para o data.yaml.
        split: Split a ser avaliado ('test' ou 'val').
        imgsz: Resolução da imagem de entrada (padrão 640).
        conf_threshold: Limiar mínimo de confiança.
        iou_threshold: Limiar de IoU para considerar detecção correta.

    Returns:
        Objeto com os resultados detalhados da validação retornado pelo Ultralytics.
    """
    print(f"Iniciando avaliação no split '{split}' (imgsz={imgsz}, conf={conf_threshold}, iou={iou_threshold})...")
    resultados = modelo.val(
        data=str(caminho_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,
        save_json=False,
        verbose=True,
    )
    return resultados


def extrair_dicionario_metricas(resultados_val, split_utilizado: str) -> dict:
    """
    Extrai as métricas numéricas do objeto de validação em um dicionário limpo.

    Por que existe:
        Desacopla a estrutura interna de dados do Ultralytics do restante da aplicação,
        garantindo um formato serializável em JSON com precisão de 4 casas decimais.

    Args:
        resultados_val: Objeto retornado por model.val().
        split_utilizado: Nome do split em que a avaliação ocorreu.

    Returns:
        Dicionário com métricas globais, tempos de latência e métricas por classe.
    """
    box = resultados_val.box
    nomes_classes = resultados_val.names

    metricas = {
        "split_avaliado": split_utilizado,
        "geral": {
            "precisao": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "velocidade_ms": {
            "preprocessamento": round(float(resultados_val.speed.get("preprocess", 0.0)), 2),
            "inferencia": round(float(resultados_val.speed.get("inference", 0.0)), 2),
            "posprocessamento": round(float(resultados_val.speed.get("postprocess", 0.0)), 2),
        },
        "classes": {}
    }

    # Extrai métricas por classe caso disponíveis
    try:
        ap50_classes = box.ap50
        ap_classes = box.ap
        p_classes = box.p
        r_classes = box.r

        for i, nome_classe in nomes_classes.items():
            metricas["classes"][nome_classe] = {
                "id": int(i),
                "precisao": round(float(p_classes[i]), 4) if len(p_classes) > i else None,
                "recall": round(float(r_classes[i]), 4) if len(r_classes) > i else None,
                "map50": round(float(ap50_classes[i]), 4) if len(ap50_classes) > i else None,
                "map50_95": round(float(ap_classes[i]), 4) if len(ap_classes) > i else None,
            }
    except Exception as e:
        print(f"[AVISO] Não foi possível detalhar métricas individuais de classe: {e}")

    return metricas


def salvar_metricas_json(metricas: dict, caminho_arquivo: Path) -> None:
    """
    Grava as métricas coletadas em um arquivo JSON.

    Por que existe:
        Cumpre a regra estrita de que toda métrica apresentada no relatório
        deve ter origem comprovada em um artefato gerado em results/.

    Args:
        metricas: Dicionário contendo as métricas calculadas.
        caminho_arquivo: Caminho completo para salvar o arquivo JSON.
    """
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    print(f"✅ Arquivo de métricas salvo em: {caminho_arquivo.resolve()}")


def copiar_artefatos_graficos(diretorio_origem: Path, diretorio_destino: Path) -> list[str]:
    """
    Copia gráficos gerados nativamente pelo Ultralytics para a pasta results/.

    Por que existe:
        O Ultralytics gera gráficos completos de curva PxR e matriz de confusão
        na pasta de runs. Esta função organiza esses artefatos dentro de results/
        com nomes padronizados para uso no relatório e no README.

    Args:
        diretorio_origem: Diretório save_dir da avaliação do Ultralytics.
        diretorio_destino: Pasta results/ do projeto.

    Returns:
        Lista de nomes de arquivos copiados com sucesso.
    """
    diretorio_destino.mkdir(parents=True, exist_ok=True)
    mapeamento_arquivos = {
        "PR_curve.png": "curva_pr.png",
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "F1_curve.png": "curva_f1.png",
        "val_batch0_pred.jpg": "amostra_predicoes_val.jpg",
    }

    copiados = []
    for nome_orig, nome_dest in mapeamento_arquivos.items():
        arquivo_origem = diretorio_origem / nome_orig
        if arquivo_origem.exists():
            destino = diretorio_destino / nome_dest
            shutil.copy(arquivo_origem, destino)
            copiados.append(nome_dest)

    print(f"✅ {len(copiados)} artefatos visuais consolidados em: {diretorio_destino.resolve()}")
    return copiados


def exibir_tabela_resumo(metricas: dict) -> None:
    """
    Exibe uma tabela formatada no terminal com as métricas obtidas.

    Por que existe:
        Facilita a conferência imediata das métricas após a execução,
        permitindo ao desenvolvedor validar o resultado sem precisar abrir o JSON.

    Args:
        metricas: Dicionário de métricas gerado por extrair_dicionario_metricas.
    """
    g = metricas["geral"]
    v = metricas["velocidade_ms"]
    split = metricas["split_avaliado"]

    print("\n" + "="*55)
    print(f"📊 RESUMO DE DESEMPENHO (Split: {split})")
    print("="*55)
    print(f"  • Precisão (Precision) : {g['precisao'] * 100:.2f}%")
    print(f"  • Recall (Sensibilidade): {g['recall'] * 100:.2f}%")
    print(f"  • mAP @ 0.50           : {g['map50'] * 100:.2f}%")
    print(f"  • mAP @ 0.50:0.95      : {g['map50_95'] * 100:.2f}%")
    print("-"*55)
    print("⚡ LATÊNCIA MÉDIA POR IMAGEM:")
    print(f"  • Pré-processamento : {v['preprocessamento']} ms")
    print(f"  • Inferência        : {v['inferencia']} ms")
    print(f"  • Pós-processamento : {v['posprocessamento']} ms")
    tempo_total = v['preprocessamento'] + v['inferencia'] + v['posprocessamento']
    fps_est = (1000.0 / tempo_total) if tempo_total > 0 else 0
    print(f"  • FPS Estimado      : ~{fps_est:.1f} frames/segundo")
    print("="*55 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Coordena o fluxo de avaliação a partir dos argumentos de linha de comando.

    Por que existe:
        Centraliza a orquestração do script evaluate.py respeitando os parâmetros
        passados pelo usuário via terminal.
    """
    parser = argparse.ArgumentParser(
        description="Avalia modelo YOLO no conjunto de teste/validação e gera metricas.json e gráficos."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Caminho para os pesos do modelo treinado (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Caminho para o arquivo data.yaml do dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split de avaliação a ser utilizado: 'test' ou 'val'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Diretório onde metricas.json e os gráficos serão salvos",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho da imagem de entrada para avaliação",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar de confiança para detecção",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="Limiar de IoU para cálculo das métricas",
    )

    args = parser.parse_args()

    caminho_pesos = Path(args.weights)
    caminho_yaml = Path(args.data)
    diretorio_saida = Path(args.output_dir)

    # 1. Carrega o modelo
    modelo = carregar_modelo_yolo(caminho_pesos)

    # 2. Valida split
    split_seguro = verificar_split_disponivel(caminho_yaml, args.split)

    # 3. Executa validação oficial
    resultados_val = executar_avaliacao(
        modelo=modelo,
        caminho_yaml=caminho_yaml,
        split=split_seguro,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # 4. Extrai dicionário de métricas
    metricas = extrair_dicionario_metricas(resultados_val, split_seguro)

    # 5. Salva results/metricas.json
    arquivo_json = diretorio_saida / "metricas.json"
    salvar_metricas_json(metricas, arquivo_json)

    # 6. Copia gráficos gerados para results/
    dir_salvo_ultralytics = Path(resultados_val.save_dir)
    copiar_artefatos_graficos(dir_salvo_ultralytics, diretorio_saida)

    # 7. Exibe resumo no terminal
    exibir_tabela_resumo(metricas)


if __name__ == "__main__":
    main()
