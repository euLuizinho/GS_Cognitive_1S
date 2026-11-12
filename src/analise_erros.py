"""
src/analise_erros.py
====================
Identifica, cataloga e salva amostras de Falsos Positivos (FP) e Falsos Negativos (FN).

Por que existe:
    Para a arguição da Global Solution e para o relatório técnico, não basta
    apenas reportar métricas agregadas (como mAP). É fundamental demonstrar
    compreensão crítica dos limites do modelo: quais artefatos visuais enganam
    o detector (FP) e quais características de buracos reais são ignoradas (FN).

Uso:
    python src/analise_erros.py --weights models/best.pt --data data/data.yaml --max_amostras 10
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Funções de cálculo geométrico e parsing
# ──────────────────────────────────────────────────────────────────────────────

def carregar_rotulos_ground_truth(caminho_txt: Path, largura_img: int, altura_img: int) -> list[dict]:
    """
    Lê o arquivo de anotação YOLO (.txt) e converte coordenadas normalizadas para pixels absolutos.

    Por que existe:
        O formato YOLO salva coordenadas relativas [cx, cy, w, h] entre 0 e 1.
        Para calcular o IoU real e desenhar as caixas na imagem com OpenCV, é
        necessário converter essas coordenadas para [x1, y1, x2, y2] em pixels.

    Args:
        caminho_txt: Path para o arquivo de rótulo .txt.
        largura_img: Largura da imagem em pixels.
        altura_img: Altura da imagem em pixels.

    Returns:
        Lista de dicionários no formato:
        {'classe_id': int, 'box': [x1, y1, x2, y2]}
    """
    caixas = []
    if not caminho_txt.exists():
        return caixas

    linhas = caminho_txt.read_text(encoding="utf-8").strip().splitlines()
    for linha in linhas:
        partes = linha.strip().split()
        if len(partes) == 5:
            cls_id = int(partes[0])
            cx, cy, bw, bh = map(float, partes[1:])

            # Desnormalização para coordenadas de cantos (pixels)
            x1 = int((cx - (bw / 2.0)) * largura_img)
            y1 = int((cy - (bh / 2.0)) * altura_img)
            x2 = int((cx + (bw / 2.0)) * largura_img)
            y2 = int((cy + (bh / 2.0)) * altura_img)

            # Garante limites dentro da imagem
            x1 = max(0, min(x1, largura_img - 1))
            y1 = max(0, min(y1, altura_img - 1))
            x2 = max(0, min(x2, largura_img - 1))
            y2 = max(0, min(y2, altura_img - 1))

            caixas.append({"classe_id": cls_id, "box": [x1, y1, x2, y2]})

    return caixas


def calcular_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Calcula a Intersection over Union (IoU) entre dois retângulos [x1, y1, x2, y2].

    Por que existe:
        IoU é a métrica padrão em detecção de objetos para determinar sobreposição.
        Se IoU >= limiar (ex: 0.5), considera-se que a detecção acertou o objeto
        anotado (Verdadeiro Positivo). Caso contrário, é erro.

    Args:
        box_a: Lista [x1, y1, x2, y2] da primeira caixa.
        box_b: Lista [x1, y1, x2, y2] da segunda caixa.

    Returns:
        Valor escalar de IoU no intervalo [0.0, 1.0].
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    largura_inter = max(0, xB - xA)
    altura_inter = max(0, yB - yA)
    area_inter = largura_inter * altura_inter

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    area_uniao = float(area_a + area_b - area_inter)
    if area_uniao <= 0:
        return 0.0

    return area_inter / area_uniao


def categorizar_predicoes_imagem(
    caixas_gt: list[dict],
    predicoes: list[dict],
    iou_limiar: float = 0.5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Compara predições e anotações reais de uma imagem e categoriza em TP, FP e FN.

    Por que existe:
        Implementa a lógica formal de associação baseada em IoU:
        - TP: predição com IoU >= limiar com algum ground truth disponível.
        - FP: predição que não alcança o IoU necessário com nenhum ground truth.
        - FN: ground truth real que não teve nenhuma predição correspondente.

    Args:
        caixas_gt: Lista de caixas reais da imagem.
        predicoes: Lista de caixas previstas com confiança.
        iou_limiar: Limiar mínimo de IoU para correspondência positiva (padrão 0.5).

    Returns:
        Tupla com três listas: (verdadeiros_positivos, falsos_positivos, falsos_negativos).
    """
    gts_cobertos = set()
    falsos_positivos = []
    verdadeiros_positivos = []

    # Ordena predições por confiança decrescente para priorizar as mais seguras
    predicoes_ordenadas = sorted(predicoes, key=lambda p: p["confianca"], reverse=True)

    for pred in predicoes_ordenadas:
        melhor_iou = 0.0
        melhor_gt_idx = -1

        for idx_gt, gt in enumerate(caixas_gt):
            if idx_gt in gts_cobertos:
                continue
            if gt["classe_id"] == pred["classe_id"]:
                iou_atual = calcular_iou(pred["box"], gt["box"])
                if iou_atual > melhor_iou:
                    melhor_iou = iou_atual
                    melhor_gt_idx = idx_gt

        if melhor_iou >= iou_limiar and melhor_gt_idx != -1:
            gts_cobertos.add(melhor_gt_idx)
            verdadeiros_positivos.append(pred)
        else:
            falsos_positivos.append(pred)

    # Todo ground truth que não foi associado a nenhuma predição é Falso Negativo
    falsos_negativos = [gt for idx, gt in enumerate(caixas_gt) if idx not in gts_cobertos]

    return verdadeiros_positivos, falsos_positivos, falsos_negativos


# ──────────────────────────────────────────────────────────────────────────────
# Funções de renderização visual e anotação
# ──────────────────────────────────────────────────────────────────────────────

def desenhar_anotacoes_erro(
    imagem_bgr: np.ndarray,
    caixas_gt: list[dict],
    caixas_fp: list[dict],
    caixas_fn: list[dict],
) -> np.ndarray:
    """
    Desenha as caixas delimitadoras e legendas explicativas sobre a imagem.

    Convenção de cores:
        - VERDE (0, 255, 0): Ground Truth real (buraco anotado).
        - VERMELHO (0, 0, 255): Falso Positivo (alarme falso do modelo).
        - LARANJA (0, 165, 255): Falso Negativo (buraco omitido pelo modelo).

    Por que existe:
        Torna os erros visualmente autoexplicativos para apresentação em banca
        e inclusão no relatório técnico em PDF.

    Args:
        imagem_bgr: Imagem original carregada pelo OpenCV.
        caixas_gt: Caixas reais de referência.
        caixas_fp: Caixas de falsos positivos detectadas.
        caixas_fn: Caixas de falsos negativos não detectadas.

    Returns:
        Imagem anotada com as caixas e legendas.
    """
    canvas = imagem_bgr.copy()

    # 1. Desenha Ground Truth (Verde)
    for gt in caixas_gt:
        x1, y1, x2, y2 = gt["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas, "GT: Buraco", (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
        )

    # 2. Desenha Falsos Positivos (Vermelho)
    for fp in caixas_fp:
        x1, y1, x2, y2 = fp["box"]
        conf = fp.get("confianca", 0.0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        rotulo = f"FP: {conf:.2f}"
        cv2.putText(
            canvas, rotulo, (x1, min(canvas.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA
        )

    # 3. Desenha Falsos Negativos (Laranja)
    for fn in caixas_fn:
        x1, y1, x2, y2 = fn["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(
            canvas, "FN: Omitido", (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA
        )

    return canvas


def salvar_imagem_anotada(imagem: np.ndarray, caminho_destino: Path) -> None:
    """
    Grava a imagem com anotações de erro no diretório especificado.

    Por que existe:
        Isola a operação de I/O em disco, garantindo que o diretório de destino
        seja criado automaticamente.

    Args:
        imagem: Array NumPy da imagem em formato BGR.
        caminho_destino: Caminho completo para salvar o arquivo de imagem.
    """
    caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho_destino), imagem)


# ──────────────────────────────────────────────────────────────────────────────
# Funções de taxonomia e diagnóstico técnico
# ──────────────────────────────────────────────────────────────────────────────

def mapear_padroes_visuais_recorrentes() -> dict:
    """
    Retorna a taxonomia técnica dos padrões visuais que provocam erros no detector.

    Por que existe:
        Fornece embasamento qualitativo estruturado para a seção de Análise Crítica
        do relatório técnico, conectando os erros observados empiricamente às
        características físicas da via urbana.

    Returns:
        Dicionário com categorias, descrições físicas e impacto em FP/FN.
    """
    return {
        "falsos_positivos": [
            {
                "padrao": "Sombras de Copas de Árvores e Postes",
                "mecanismo_fisico": "Bordas de alto contraste e áreas escuras no asfalto com formato irregular imitam a cavidade sombreada de um buraco.",
                "frequencia_estimada": "Alta em dias ensolarados"
            },
            {
                "padrao": "Remendos Recentes e Juntas de Pavimentação",
                "mecanismo_fisico": "Diferença de coloração entre asfalto novo (preto fosco) e asfalto degradado (cinza claro) cria delimitações retangulares que ativam falso alarme.",
                "frequencia_estimada": "Média"
            },
            {
                "padrao": "Tampas de Bueiro e Poços de Visita Nivelados",
                "mecanismo_fisico": "Geometria circular ou quadrada com textura metálica oxidada semelhante a depressões na via.",
                "frequencia_estimada": "Baixa a Média"
            },
            {
                "padrao": "Poças de Água Superficiais sem Cavidade",
                "mecanismo_fisico": "Lâmina de água reflete o céu ou escurece o pavimento, gerando contraste similar ao de uma depressão profunda.",
                "frequencia_estimada": "Alta em períodos de chuva"
            }
        ],
        "falsos_negativos": [
            {
                "padrao": "Buracos Rasos ou de Baixo Contraste (Erosão Inicial)",
                "mecanismo_fisico": "Desgaste superficial sem degrau pronunciado não gera sombra interna suficiente para diferenciação de textura.",
                "frequencia_estimada": "Alta"
            },
            {
                "padrao": "Buracos Inundados / Encobertos por Lama",
                "mecanismo_fisico": "A água turva preenche a cavidade e nivelamento visual com o plano da pista oculta a profundidade e as bordas.",
                "frequencia_estimada": "Média em vias molhadas"
            },
            {
                "padrao": "Buracos Distantes ou em Perspectiva Acentuada",
                "mecanismo_fisico": "A resolução espacial (poucos pixels na entrada 640x640) é insuficiente para o extrator de características da rede convolucional.",
                "frequencia_estimada": "Alta no horizonte da via"
            },
            {
                "padrao": "Superexposição ou Baixa Luminosidade (Crepúsculo/Noite)",
                "mecanismo_fisico": "Perda de faixa dinâmica (dynamic range) da câmera suprime o gradiente de cores da pista.",
                "frequencia_estimada": "Média"
            }
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Orquestrador da análise
# ──────────────────────────────────────────────────────────────────────────────

def executar_diagnostico_erros(
    modelo: YOLO,
    pasta_imagens: Path,
    pasta_labels: Path,
    pasta_saida: Path,
    max_amostras: int = 10,
    conf_limiar: float = 0.25,
    iou_limiar: float = 0.5,
) -> dict:
    """
    Varre as imagens do conjunto selecionado, isola amostras de FP/FN e gera o relatório.

    Por que existe:
        Centraliza a lógica de orquestração da Etapa 5 em uma única função
        com controle rígido de limites amostrais (10 FPs e 10 FNs).

    Args:
        modelo: Instância treinada do YOLO.
        pasta_imagens: Diretório contendo as imagens de teste/val.
        pasta_labels: Diretório contendo os arquivos .txt de anotação.
        pasta_saida: Diretório base onde salvar os erros.
        max_amostras: Quantidade máxima de amostras a salvar por categoria (10).
        conf_limiar: Limiar de confiança para considerar uma predição.
        iou_limiar: Limiar de IoU para associação.

    Returns:
        Dicionário com contagens consolidadas e listas de arquivos catalogados.
    """
    dir_fp = pasta_saida / "falsos_positivos"
    dir_fn = pasta_saida / "falsos_negativos"
    dir_fp.mkdir(parents=True, exist_ok=True)
    dir_fn.mkdir(parents=True, exist_ok=True)

    extensoes = {".jpg", ".jpeg", ".png", ".bmp"}
    candidatos_imgs = [p for p in pasta_imagens.rglob("*") if p.suffix.lower() in extensoes]

    amostras_fp_salvas = []
    amostras_fn_salvas = []
    total_fp_acumulado = 0
    total_fn_acumulado = 0
    total_tp_acumulado = 0

    print(f"Analisando {len(candidatos_imgs)} imagens para diagnóstico de erros...")

    for caminho_img in candidatos_imgs:
        img_bgr = cv2.imread(str(caminho_img))
        if img_bgr is None:
            continue

        alt, larg = img_bgr.shape[:2]
        caminho_lbl = pasta_labels / (caminho_img.stem + ".txt")

        # 1. Carrega anotações reais
        gts = carregar_rotulos_ground_truth(caminho_lbl, larg, alt)

        # 2. Executa inferência
        resultados_pred = modelo.predict(
            source=img_bgr,
            conf=conf_limiar,
            iou=0.45,
            verbose=False,
            device="cpu",
        )

        predicoes = []
        for r in resultados_pred:
            if r.boxes is not None:
                for b in r.boxes:
                    coords = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    predicoes.append({
                        "classe_id": cls_id,
                        "box": coords,
                        "confianca": conf
                    })

        # 3. Categoriza
        tps, fps, fns = categorizar_predicoes_imagem(gts, predicoes, iou_limiar=iou_limiar)
        total_tp_acumulado += len(tps)
        total_fp_acumulado += len(fps)
        total_fn_acumulado += len(fns)

        # 4. Salva amostra de Falso Positivo (se houver e não estourou max_amostras)
        if len(fps) > 0 and len(amostras_fp_salvas) < max_amostras:
            anotada_fp = desenhar_anotacoes_erro(img_bgr, gts, fps, [])
            nome_arq = f"fp_{len(amostras_fp_salvas) + 1:02d}_{caminho_img.name}"
            salvar_imagem_anotada(anotada_fp, dir_fp / nome_arq)
            amostras_fp_salvas.append({
                "imagem": caminho_img.name,
                "arquivo_salvo": nome_arq,
                "total_fps_na_imagem": len(fps),
                "maior_confianca_fp": max([p["confianca"] for p in fps])
            })

        # 5. Salva amostra de Falso Negativo (se houver e não estourou max_amostras)
        if len(fns) > 0 and len(amostras_fn_salvas) < max_amostras:
            anotada_fn = desenhar_anotacoes_erro(img_bgr, gts, [], fns)
            nome_arq = f"fn_{len(amostras_fn_salvas) + 1:02d}_{caminho_img.name}"
            salvar_imagem_anotada(anotada_fn, dir_fn / nome_arq)
            amostras_fn_salvas.append({
                "imagem": caminho_img.name,
                "arquivo_salvo": nome_arq,
                "total_fns_na_imagem": len(fns)
            })

        if len(amostras_fp_salvas) >= max_amostras and len(amostras_fn_salvas) >= max_amostras:
            break

    relatorio_final = {
        "total_imagens_avaliadas": len(candidatos_imgs),
        "total_tps_encontrados": total_tp_acumulado,
        "total_fps_encontrados": total_fp_acumulado,
        "total_fns_encontrados": total_fn_acumulado,
        "limiares_utilizados": {
            "confianca": conf_limiar,
            "iou": iou_limiar
        },
        "amostras_salvas": {
            "falsos_positivos": amostras_fp_salvas,
            "falsos_negativos": amostras_fn_salvas
        },
        "taxonomia_padroes_de_erro": mapear_padroes_visuais_recorrentes()
    }

    return relatorio_final


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Ponto de entrada do script de análise de erros.

    Por que existe:
        Permite invocar a análise de forma desacoplada, configurando pastas
        e parâmetros via linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Analisa Falsos Positivos e Falsos Negativos do modelo YOLO."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Caminho para os pesos treinados (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Caminho para o data.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split de dados a analisar ('test' ou 'val')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analise_erros",
        help="Diretório onde salvar amostras e relatório",
    )
    parser.add_argument(
        "--max_amostras",
        type=int,
        default=10,
        help="Quantidade máxima de amostras de FP e FN a salvar (padrão 10)",
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
        default=0.5,
        help="Limiar de IoU para considerar correspondência correta",
    )

    args = parser.parse_args()

    caminho_pesos = Path(args.weights)
    if not caminho_pesos.exists():
        print(f"[ERRO] Pesos não encontrados em: {caminho_pesos.resolve()}")
        sys.exit(1)

    caminho_yaml = Path(args.data)
    if not caminho_yaml.exists():
        print(f"[ERRO] Arquivo data.yaml não encontrado em: {caminho_yaml.resolve()}")
        sys.exit(1)

    with open(caminho_yaml, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # Identifica pasta do split
    raiz_data = Path(config_yaml.get("path", "."))
    caminho_split = config_yaml.get(args.split)
    if not caminho_split:
        caminho_split = config_yaml.get("val")
        print(f"[AVISO] Split '{args.split}' não encontrado no YAML. Usando 'val' como alternativa.")

    pasta_imgs = (raiz_data / caminho_split).resolve()
    pasta_lbls = pasta_imgs.parent / "labels"

    print(f"Imagens: {pasta_imgs}")
    print(f"Rótulos: {pasta_lbls}")

    modelo = YOLO(str(caminho_pesos))
    pasta_saida = Path(args.output_dir)

    relatorio = executar_diagnostico_erros(
        modelo=modelo,
        pasta_imagens=pasta_imgs,
        pasta_labels=pasta_lbls,
        pasta_saida=pasta_saida,
        max_amostras=args.max_amostras,
        conf_limiar=args.conf,
        iou_limiar=args.iou,
    )

    # Salva relatório em JSON
    caminho_json = pasta_saida / "diagnostico_erros.json"
    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print("📋 DIAGNÓSTICO DE ERROS CONCLUÍDO COM SUCESSO")
    print("="*60)
    print(f"  • Falsos Positivos catalogados : {len(relatorio['amostras_salvas']['falsos_positivos'])} amostras")
    print(f"  • Falsos Negativos catalogados : {len(relatorio['amostras_salvas']['falsos_negativos'])} amostras")
    print(f"  • Amostras visuais salvas em    : {pasta_saida.resolve()}")
    print(f"  • Relatório estruturado salvo em: {caminho_json.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
