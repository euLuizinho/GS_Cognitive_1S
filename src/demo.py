"""
src/demo.py
===========
Aplicação de demonstração visual em tempo real para detecção de buracos em vias urbanas.

Por que existe:
    Atende diretamente ao requisito de "Apresentação Presencial / Demonstração Prática"
    da Global Solution (peso de 1,5 pontos na nota final). Permite ao aluno rodar
    a inferência ao vivo na frente do professor no computador do laboratório,
    aceitando imagem estática, arquivo de vídeo gravado ou webcam, exibindo
    bounding boxes, contador de buracos e taxa de quadros por segundo (FPS).

Uso:
    # 1. Imagem estática:
    python src/demo.py --source examples/buraco.jpg --weights models/best.pt

    # 2. Arquivo de vídeo:
    python src/demo.py --source examples/transito.mp4 --weights models/best.pt

    # 3. Webcam do computador (câmera 0):
    python src/demo.py --source 0 --weights models/best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Funções com responsabilidade única
# ──────────────────────────────────────────────────────────────────────────────

def carregar_detector_yolo(caminho_pesos: Path, device: str | None = None) -> YOLO:
    """
    Carrega o modelo YOLO a partir dos pesos fornecidos e aloca no hardware disponível.

    Por que existe:
        Garante portabilidade: se estiver no laboratório sem placa gráfica dedicada,
        o modelo roda automaticamente em CPU; se houver placa NVIDIA, usa CUDA
        para máxima taxa de FPS.

    Args:
        caminho_pesos: Path para o arquivo .pt (ex: models/best.pt).
        device: Dispositivo explícito ('cpu', 'cuda:0' ou None para autodetecção).

    Returns:
        Instância do modelo YOLO pronta para inferência.
    """
    if not caminho_pesos.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos não encontrado em: {caminho_pesos.resolve()}.\n"
            f"Certifique-se de baixar o 'best.pt' para a pasta models/."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Carregando YOLOv8 de: {caminho_pesos.name} | Dispositivo: {device.upper()}")
    modelo = YOLO(str(caminho_pesos))
    modelo.to(device)
    return modelo


def determinar_tipo_fonte(fonte_str: str) -> tuple[str, str | int]:
    """
    Identifica se a fonte de entrada é uma webcam, um vídeo gravado ou uma foto.

    Por que existe:
        Evita duplicação de scripts: um único ponto de entrada atende aos três
        modos de apresentação exigidos na banca com base no argumento passado.

    Args:
        fonte_str: String passada pelo usuário (ex: '0', 'video.mp4', 'foto.jpg').

    Returns:
        Tupla com (tipo_fonte, fonte_processada), onde tipo_fonte é
        'webcam', 'video' ou 'imagem'.
    """
    # Se for dígito numérico puro, trata como índice de webcam
    if fonte_str.isdigit():
        return "webcam", int(fonte_str)

    caminho = Path(fonte_str)
    extensoes_img = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    extensoes_vid = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    if caminho.suffix.lower() in extensoes_img:
        return "imagem", str(caminho)
    elif caminho.suffix.lower() in extensoes_vid:
        return "video", str(caminho)
    else:
        # Fallback tenta como caminho genérico
        return "imagem", str(caminho)


def executar_inferencia_frame(
    modelo: YOLO,
    frame: np.ndarray,
    conf_limiar: float = 0.25,
    iou_limiar: float = 0.45,
    imgsz: int = 640,
) -> tuple[list[dict], float]:
    """
    Executa a predição do modelo em um único quadro de imagem e mede a latência.

    Por que existe:
        Isola a operação de inferência da renderização gráfica, permitindo
        cronometrar com precisão milissegundos de forward pass sem incluir o tempo
        de desenho do OpenCV.

    Args:
        modelo: Modelo YOLO carregado.
        frame: Imagem em formato BGR (OpenCV).
        conf_limiar: Limiar de confiança mínima.
        iou_limiar: Limiar de IoU para Non-Maximum Suppression (NMS).
        imgsz: Tamanho da resolução de entrada da rede.

    Returns:
        Tupla contendo:
        - Lista de detecções: [{'box': [x1, y1, x2, y2], 'conf': float, 'cls': int, 'nome': str}]
        - Tempo de inferência em milissegundos.
    """
    inicio = time.perf_counter()

    resultados = modelo.predict(
        source=frame,
        conf=conf_limiar,
        iou=iou_limiar,
        imgsz=imgsz,
        verbose=False,
    )

    duracao_ms = (time.perf_counter() - inicio) * 1000.0

    deteccoes = []
    nomes = modelo.names

    for r in resultados:
        if r.boxes is not None:
            for b in r.boxes:
                coords = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                nome_classe = nomes.get(cls_id, f"classe_{cls_id}")
                deteccoes.append({
                    "box": coords,
                    "conf": conf,
                    "cls": cls_id,
                    "nome": nome_classe,
                })

    return deteccoes, duracao_ms


def desenhar_deteccoes_frame(frame: np.ndarray, deteccoes: list[dict]) -> np.ndarray:
    """
    Desenha os retângulos de bounding box e os rótulos de classe/confiança.

    Por que existe:
        Garante legibilidade na projeção da apresentação: caixas com contraste
        adequado, bordas arredondadas e fundo sólido no texto para leitura fácil.

    Args:
        frame: Quadro original em formato BGR.
        deteccoes: Lista de detecções retornadas por executar_inferencia_frame.

    Returns:
        Frame anotado com os retângulos de detecção.
    """
    canvas = frame.copy()

    for det in deteccoes:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        nome = det["nome"]

        # Cor ciano/azul destacada para os buracos (formato BGR: Azul=255, Verde=180, Vermelho=0)
        cor_bbox = (255, 180, 0)

        # 1. Bounding box principal
        cv2.rectangle(canvas, (x1, y1), (x2, y2), cor_bbox, 2)

        # 2. Rótulo com fundo preenchido para contraste
        texto = f"{nome}: {conf:.2f}"
        tam_texto, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        largura_txt, altura_txt = tam_texto

        # Fundo do texto
        cv2.rectangle(
            canvas,
            (x1, max(0, y1 - altura_txt - 8)),
            (x1 + largura_txt + 8, y1),
            cor_bbox,
            -1
        )

        # Texto em preto sobre fundo ciano
        cv2.putText(
            canvas,
            texto,
            (x1 + 4, max(altura_txt + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def desenhar_hud_informacoes(
    frame: np.ndarray,
    total_buracos: int,
    fps: float,
    latencia_ms: float,
) -> np.ndarray:
    """
    Desenha uma barra superior semi-transparente (HUD) com métricas em tempo real.

    Por que existe:
        Atende à exigência explícita do enunciado da matéria: exibir na tela
        a contagem de buracos identificados e o FPS em tempo real durante a demo.

    Args:
        frame: Quadro anotado com detecções.
        total_buracos: Quantidade total de buracos detectados no quadro.
        fps: Taxa calculada de quadros por segundo.
        latencia_ms: Tempo de inferência do modelo no quadro em milissegundos.

    Returns:
        Frame com a barra superior de informações desenhada.
    """
    altura, largura = frame.shape[:2]
    altura_barra = 42

    # Cria camada translúcida escura no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (largura, altura_barra), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Informações textuais
    texto_contagem = f"Buracos Detectados: {total_buracos}"
    texto_fps = f"FPS: {fps:.1f} ({latencia_ms:.1f} ms)"
    texto_projeto = "FIAP GS 2026.2 | CitizenTech"

    # Cores (BGR)
    cor_alerta = (0, 165, 255) if total_buracos > 0 else (0, 255, 0)
    cor_branca = (240, 240, 240)

    # 1. Contagem à esquerda
    cv2.putText(
        frame, texto_contagem, (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor_alerta, 2, cv2.LINE_AA
    )

    # 2. FPS no centro/direita
    cv2.putText(
        frame, texto_fps, (largura - 300, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_branca, 1, cv2.LINE_AA
    )

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Controladores de fluxo (Imagem vs Vídeo/Webcam)
# ──────────────────────────────────────────────────────────────────────────────

def processar_modo_imagem(
    modelo: YOLO,
    caminho_imagem: str,
    conf_limiar: float,
    iou_limiar: float,
    caminho_saida: Path | None = None,
) -> None:
    """
    Carrega uma imagem estática, aplica detecção e exibe na janela OpenCV.

    Por que existe:
        Permite testar imagens individuais rapidamente no computador do laboratório
        sem depender de câmera funcional ou arquivo de vídeo grande.

    Args:
        modelo: Instância carregada do YOLO.
        caminho_imagem: Caminho do arquivo de imagem.
        conf_limiar: Limiar de confiança.
        iou_limiar: Limiar de IoU para NMS.
        caminho_saida: Opcional para gravar a imagem anotada em disco.
    """
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"[ERRO] Não foi possível ler a imagem em: {caminho_imagem}")
        return

    deteccoes, latencia_ms = executar_inferencia_frame(
        modelo, img, conf_limiar=conf_limiar, iou_limiar=iou_limiar
    )

    fps_estatico = (1000.0 / latencia_ms) if latencia_ms > 0 else 0.0

    frame_anotado = desenhar_deteccoes_frame(img, deteccoes)
    frame_final = desenhar_hud_informacoes(
        frame_anotado, len(deteccoes), fps_estatico, latencia_ms
    )

    if caminho_saida:
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(caminho_saida), frame_final)
        print(f"✅ Resultado salvo em: {caminho_saida.resolve()}")

    print(f"\n[INFO] Buracos detectados: {len(deteccoes)} | Tempo: {latencia_ms:.1f} ms")
    print("[INFO] Pressione qualquer tecla na janela para fechar.")

    cv2.imshow("Deteccao de Buracos - FIAP GS 2026.2", frame_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def processar_modo_stream(
    modelo: YOLO,
    fonte: str | int,
    conf_limiar: float,
    iou_limiar: float,
    caminho_saida: Path | None = None,
) -> None:
    """
    Processa fluxo contínuo de vídeo gravado ou captura ao vivo da webcam.

    Por que existe:
        Implementa o loop principal de captura de vídeo com medição precisa de FPS,
        processamento quadro a quadro e suporte a encerramento gracioso via tecla 'q'.

    Args:
        modelo: Instância carregada do YOLO.
        fonte: Índice numérico da câmera ou caminho para arquivo .mp4/.avi.
        conf_limiar: Limiar de confiança.
        iou_limiar: Limiar de IoU para NMS.
        caminho_saida: Caminho opcional para exportar o vídeo resultante processado.
    """
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"[ERRO] Falha ao abrir fonte de vídeo/webcam: {fonte}")
        return

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 30.0

    gravador = None
    if caminho_saida:
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        gravador = cv2.VideoWriter(str(caminho_saida), fourcc, fps_fonte, (largura, altura))

    print("\n" + "="*55)
    print("🎥 DEMONSTRAÇÃO EM EXECUÇÃO")
    print("   • Tecla 'q': Encerrar demonstração")
    print("   • Tecla 's': Salvar captura do frame atual")
    print("="*55 + "\n")

    t_anterior = time.perf_counter()
    fps_suavizado = 0.0
    num_screenshot = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fim do arquivo de vídeo ou câmera desconectada.")
                break

            # 1. Inferência no quadro
            deteccoes, latencia_ms = executar_inferencia_frame(
                modelo, frame, conf_limiar=conf_limiar, iou_limiar=iou_limiar
            )

            # 2. Cálculo do FPS real do pipeline
            t_atual = time.perf_counter()
            delta_t = t_atual - t_anterior
            t_anterior = t_atual
            fps_instantaneo = (1.0 / delta_t) if delta_t > 0 else 0.0

            # Suavização exponencial para estabilizar leitura do FPS na tela
            fps_suavizado = 0.9 * fps_suavizado + 0.1 * fps_instantaneo if fps_suavizado > 0 else fps_instantaneo

            # 3. Desenho de anotações e painel de informações
            frame_anotado = desenhar_deteccoes_frame(frame, deteccoes)
            frame_final = desenhar_hud_informacoes(
                frame_anotado, len(deteccoes), fps_suavizado, latencia_ms
            )

            # 4. Grava se configurado
            if gravador is not None:
                gravador.write(frame_final)

            # 5. Exibição na tela
            cv2.imshow("Deteccao de Buracos - FIAP GS 2026.2 (Pressione 'q' para sair)", frame_final)

            # 6. Captura de teclas
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                print("[INFO] Encerrando por solicitação do usuário.")
                break
            elif tecla == ord("s"):
                num_screenshot += 1
                arq_screenshot = Path(f"results/screenshot_{num_screenshot:02d}.jpg")
                arq_screenshot.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(arq_screenshot), frame_final)
                print(f"📸 Screenshot salva em: {arq_screenshot.resolve()}")

    finally:
        cap.release()
        if gravador is not None:
            gravador.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Ponto de entrada do script demo.py.

    Por que existe:
        Gerencia os parâmetros de linha de comando e direciona o fluxo para
        o modo adequado de visualização (imagem ou vídeo/webcam).
    """
    parser = argparse.ArgumentParser(
        description="Demonstração visual do detector de buracos YOLOv8 em tempo real."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fonte de entrada: '0' para webcam, caminho para foto (.jpg/.png) ou vídeo (.mp4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Caminho para os pesos do modelo YOLOv8 (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar de confiança para detecção (padrão 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Limiar de IoU para NMS (padrão 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução ('cpu' ou 'cuda'). Padrão: automático",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho opcional para salvar o resultado (.jpg ou .mp4)",
    )

    args = parser.parse_args()

    caminho_pesos = Path(args.weights)
    modelo = carregar_detector_yolo(caminho_pesos, device=args.device)

    tipo_fonte, fonte_processada = determinar_tipo_fonte(args.source)
    caminho_saida = Path(args.output) if args.output else None

    if tipo_fonte == "imagem":
        processar_modo_imagem(
            modelo=modelo,
            caminho_imagem=str(fonte_processada),
            conf_limiar=args.conf,
            iou_limiar=args.iou,
            caminho_saida=caminho_saida,
        )
    else:
        processar_modo_stream(
            modelo=modelo,
            fonte=fonte_processada,
            conf_limiar=args.conf,
            iou_limiar=args.iou,
            caminho_saida=caminho_saida,
        )


if __name__ == "__main__":
    main()
