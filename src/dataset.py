"""
src/dataset.py
==============
Valida a estrutura de um dataset no formato YOLO e gera o arquivo data.yaml.

Por que existe:
    O Ultralytics exige um data.yaml bem-formado para treinar e avaliar.
    Este script verifica se as pastas e anotações estão corretas ANTES de
    subir o treino no Colab, evitando falhas a meio caminho.

Uso:
    python src/dataset.py --data_dir data/ --output data/data.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────────────────────────────────────

def listar_imagens(pasta: Path) -> list[Path]:
    """
    Retorna lista de imagens válidas (jpg, jpeg, png, bmp) dentro de uma pasta.

    Por que existe:
        Centraliza a lógica de extensões aceitas pelo Ultralytics para não
        repetir o filtro em vários lugares do código.

    Args:
        pasta: Caminho da pasta a inspecionar.

    Returns:
        Lista de Path de imagens encontradas.
    """
    extensoes = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in pasta.rglob("*") if p.suffix.lower() in extensoes]


def listar_anotacoes(pasta: Path) -> list[Path]:
    """
    Retorna lista de arquivos de anotação YOLO (.txt) dentro de uma pasta.

    Por que existe:
        Separa a busca por imagens da busca por rótulos, tornando as
        verificações independentes e mais fáceis de depurar.

    Args:
        pasta: Caminho da pasta a inspecionar (geralmente labels/).

    Returns:
        Lista de Path de arquivos .txt encontrados.
    """
    return list(pasta.rglob("*.txt"))


def verificar_par_imagem_rotulo(img_dir: Path, lbl_dir: Path) -> tuple[int, list[str]]:
    """
    Verifica se cada imagem possui um arquivo de rótulo correspondente.

    Por que existe:
        Imagens sem rótulo são tratadas como fundo pelo Ultralytics (pode ser
        intencional), mas imagens com rótulo ausente por erro causam métricas
        enganosas. Este função detecta e reporta as discrepâncias.

    Args:
        img_dir: Pasta contendo as imagens.
        lbl_dir: Pasta contendo os arquivos .txt de anotação.

    Returns:
        Tupla (quantidade de pares válidos, lista de avisos de discrepância).
    """
    imagens = listar_imagens(img_dir)
    avisos = []
    pares_validos = 0

    for img in imagens:
        rotulo = lbl_dir / (img.stem + ".txt")
        if rotulo.exists():
            pares_validos += 1
        else:
            avisos.append(f"  [SEM RÓTULO] {img.relative_to(img_dir.parent)}")

    return pares_validos, avisos


def validar_formato_rotulo(arquivo_txt: Path) -> list[str]:
    """
    Verifica se cada linha de um arquivo de rótulo YOLO tem o formato correto.

    Formato esperado por linha: <class_id> <cx> <cy> <w> <h>
    Todos os valores numéricos; cx, cy, w, h normalizados em [0.0, 1.0].

    Por que existe:
        Rótulos malformados causam erros silenciosos no treino. Validar antes
        poupa horas de debugging no Colab.

    Args:
        arquivo_txt: Caminho do arquivo de rótulo a verificar.

    Returns:
        Lista de strings descrevendo erros encontrados (vazia se tudo OK).
    """
    erros = []
    try:
        linhas = arquivo_txt.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return [f"  [LEITURA] {arquivo_txt.name}: {e}"]

    for i, linha in enumerate(linhas, 1):
        partes = linha.strip().split()
        if len(partes) != 5:
            erros.append(
                f"  [FORMATO] {arquivo_txt.name} linha {i}: "
                f"esperado 5 campos, encontrado {len(partes)}"
            )
            continue
        try:
            class_id = int(partes[0])
            coords = [float(x) for x in partes[1:]]
        except ValueError:
            erros.append(
                f"  [TIPO] {arquivo_txt.name} linha {i}: "
                f"valores não numéricos → '{linha}'"
            )
            continue
        for j, v in enumerate(coords):
            if not (0.0 <= v <= 1.0):
                erros.append(
                    f"  [RANGE] {arquivo_txt.name} linha {i}: "
                    f"coord[{j}]={v:.4f} fora de [0,1]"
                )
    return erros


def detectar_classes(lbl_dir: Path) -> list[int]:
    """
    Lê todos os rótulos e retorna a lista ordenada de IDs de classe presentes.

    Por que existe:
        Garante que o data.yaml seja gerado com exatamente as classes que
        existem no dataset — nem mais, nem menos.

    Args:
        lbl_dir: Pasta contendo os arquivos .txt de anotação.

    Returns:
        Lista ordenada de inteiros representando os IDs de classe encontrados.
    """
    ids = set()
    for txt in listar_anotacoes(lbl_dir):
        for linha in txt.read_text(encoding="utf-8").strip().splitlines():
            partes = linha.strip().split()
            if partes:
                try:
                    ids.add(int(partes[0]))
                except ValueError:
                    pass
    return sorted(ids)


def validar_estrutura_dataset(data_dir: Path) -> dict:
    """
    Valida a estrutura completa de um dataset YOLO e retorna um relatório.

    Estrutura esperada:
        data_dir/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/   (ou val/)
        │   ├── images/
        │   └── labels/
        └── test/    (opcional)
            ├── images/
            └── labels/

    Por que existe:
        Ponto de entrada único para toda a validação. Chama as funções
        menores e consolida os resultados, seguindo o princípio de uma
        responsabilidade por função.

    Args:
        data_dir: Raiz do dataset.

    Returns:
        Dicionário com chaves 'splits' (estatísticas por split),
        'erros' (lista de erros críticos) e 'avisos' (lista de avisos).
    """
    relatorio = {"splits": {}, "erros": [], "avisos": []}

    # Aceita 'valid' ou 'val' como nome do split de validação
    splits_candidatos = {
        "train": data_dir / "train",
        "valid": data_dir / "valid" if (data_dir / "valid").exists() else data_dir / "val",
        "test":  data_dir / "test",
    }

    for nome, pasta_split in splits_candidatos.items():
        if not pasta_split.exists():
            if nome == "train":
                relatorio["erros"].append(f"[CRÍTICO] Pasta '{nome}/' não encontrada em {data_dir}")
            elif nome == "valid":
                relatorio["erros"].append(f"[CRÍTICO] Pasta 'valid/' ou 'val/' não encontrada em {data_dir}")
            # test é opcional
            continue

        img_dir = pasta_split / "images"
        lbl_dir = pasta_split / "labels"

        if not img_dir.exists():
            relatorio["erros"].append(f"[CRÍTICO] {nome}/images/ não existe")
            continue
        if not lbl_dir.exists():
            relatorio["erros"].append(f"[CRÍTICO] {nome}/labels/ não existe")
            continue

        imagens = listar_imagens(img_dir)
        pares, avisos_pares = verificar_par_imagem_rotulo(img_dir, lbl_dir)
        relatorio["avisos"].extend(avisos_pares)

        # Valida formato de até 50 rótulos por split (amostragem)
        txts = listar_anotacoes(lbl_dir)
        for txt in txts[:50]:
            erros_fmt = validar_formato_rotulo(txt)
            relatorio["erros"].extend(erros_fmt)

        relatorio["splits"][nome] = {
            "pasta":        str(pasta_split),
            "imagens":      len(imagens),
            "rotulos":      len(txts),
            "pares_validos": pares,
        }

    return relatorio


def gerar_data_yaml(
    data_dir: Path,
    classes: list[str],
    output: Path,
    splits: dict,
) -> None:
    """
    Gera o arquivo data.yaml no formato exigido pelo Ultralytics.

    Por que existe:
        O data.yaml é o contrato entre o dataset e o framework de treino.
        Gerá-lo programaticamente evita erros de digitação nos caminhos,
        que são uma causa comum de falha silenciosa no treino.

    Args:
        data_dir: Raiz absoluta do dataset.
        classes:  Lista de nomes de classe na ordem dos IDs (ex: ['pothole']).
        output:   Caminho onde o arquivo será salvo.
        splits:   Dicionário com os splits detectados (para ajustar os paths).
    """
    # Detecta o nome correto do split de validação (valid ou val)
    valid_path = data_dir / "valid" if (data_dir / "valid").exists() else data_dir / "val"

    conteudo = {
        "path":  str(data_dir.resolve()),
        "train": "train/images",
        "val":   f"{valid_path.name}/images",
        "nc":    len(classes),
        "names": classes,
    }

    # Adiciona test apenas se existir
    if "test" in splits:
        conteudo["test"] = "test/images"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(conteudo, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] data.yaml gerado em: {output}")


def imprimir_relatorio(relatorio: dict) -> None:
    """
    Imprime o relatório de validação no terminal de forma legível.

    Por que existe:
        Separar a lógica de impressão da lógica de validação permite
        redirecionar a saída (para arquivo de log) sem alterar o código
        de validação.

    Args:
        relatorio: Dicionário retornado por validar_estrutura_dataset().
    """
    print("\n" + "="*60)
    print("  RELATÓRIO DE VALIDAÇÃO DO DATASET")
    print("="*60)

    for split, stats in relatorio["splits"].items():
        print(f"\n📁 Split: {split}")
        print(f"   Imagens:       {stats['imagens']}")
        print(f"   Rótulos:       {stats['rotulos']}")
        print(f"   Pares válidos: {stats['pares_validos']}")

    if relatorio["erros"]:
        print(f"\n❌ ERROS ({len(relatorio['erros'])}):")
        for e in relatorio["erros"]:
            print(e)
    else:
        print("\n✅ Nenhum erro crítico encontrado.")

    if relatorio["avisos"]:
        print(f"\n⚠️  AVISOS ({len(relatorio['avisos'])}):")
        for a in relatorio["avisos"][:20]:   # limita a 20 para não poluir
            print(a)
        if len(relatorio["avisos"]) > 20:
            print(f"   ... e mais {len(relatorio['avisos']) - 20} avisos.")
    else:
        print("✅ Nenhum aviso.")

    print("="*60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Ponto de entrada do script: lê argumentos, valida e gera o data.yaml.

    Por que existe:
        Mantém a lógica de CLI separada das funções de negócio, permitindo
        importar as funções em outros scripts sem efeitos colaterais.
    """
    parser = argparse.ArgumentParser(
        description="Valida dataset YOLO e gera data.yaml"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Raiz do dataset (deve conter train/, valid/ e test/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/data.yaml",
        help="Caminho de saída do data.yaml",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["pothole"],
        help="Nomes das classes na ordem dos IDs (ex: --classes pothole crack)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output   = Path(args.output)

    if not data_dir.exists():
        print(f"[ERRO] Diretório não encontrado: {data_dir}")
        sys.exit(1)

    print(f"Validando dataset em: {data_dir.resolve()}")

    relatorio = validar_estrutura_dataset(data_dir)
    imprimir_relatorio(relatorio)

    if relatorio["erros"]:
        print("[ABORTADO] Corrija os erros antes de gerar o data.yaml.")
        sys.exit(1)

    # Detecta classes automaticamente se não informadas pelo usuário
    lbl_dirs = []
    for split in ["train", "valid", "val"]:
        d = data_dir / split / "labels"
        if d.exists():
            lbl_dirs.append(d)

    ids_detectados = set()
    for d in lbl_dirs:
        ids_detectados.update(detectar_classes(d))

    if len(ids_detectados) > len(args.classes):
        print(
            f"[AVISO] {len(ids_detectados)} classes detectadas nos rótulos, "
            f"mas apenas {len(args.classes)} informadas via --classes. "
            f"IDs encontrados: {sorted(ids_detectados)}"
        )

    gerar_data_yaml(data_dir, args.classes, output, relatorio["splits"])
    print("Pronto! Próximo passo: abra notebooks/train_colab.ipynb no Colab.")


if __name__ == "__main__":
    main()
