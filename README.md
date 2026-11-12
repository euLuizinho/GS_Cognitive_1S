# Detecção de Buracos em Vias Urbanas com YOLO
### Global Solution 2026.2 — FIAP | Cognitive Computing | CitizenTech

> **Status:** 🚧 Em construção — Etapa 1 de 8

---

## Sumário
1. [Sobre o Projeto](#sobre-o-projeto)
2. [Alinhamento ODS e CitizenTech](#alinhamento-ods-e-citizentech)
3. [Estrutura do Repositório](#estrutura-do-repositório)
4. [Requisitos e Instalação](#requisitos-e-instalação)
5. [Execução Rápida](#execução-rápida)
6. [Treino no Google Colab](#treino-no-google-colab)
7. [Avaliação e Métricas](#avaliação-e-métricas)
8. [Demo (imagem / vídeo / webcam)](#demo)
9. [Resultados Obtidos](#resultados-obtidos)
10. [Referências](#referências)

---

## Sobre o Projeto

Buracos (potholes) em vias urbanas causam **acidentes, danos a veículos e prejuízos econômicos** bilionários anualmente no Brasil. A zeladoria pública depende de relatos manuais, lentos e geograficamente desiguais.

Este projeto treina um modelo **YOLOv8** para detectar automaticamente buracos em imagens e vídeos urbanos. O sistema funciona como **motor de visão computacional** para aplicativos de zeladoria colaborativa: o cidadão registra um vídeo pelo celular, o modelo detecta e conta os buracos em tempo real, e as coordenadas são enviadas para uma plataforma de gestão pública.

**Tecnologias:** Python · Ultralytics YOLOv8 · OpenCV · Google Colab (GPU T4)

---

## Alinhamento ODS e CitizenTech

| ODS | Meta | Como este projeto contribui |
|-----|------|-----------------------------|
| **ODS 11** — Cidades e Comunidades Sustentáveis | 11.2 Acesso a sistemas de transporte seguros | Detecta automaticamente danos em vias, priorizando manutenção |
| **ODS 9** — Indústria, Inovação e Infraestrutura | 9.1 Infraestrutura resiliente e sustentável | Visão computacional de baixo custo para monitoramento de infraestrutura |

---

## Estrutura do Repositório

```
pothole-detection-gs/
│
├── src/                    # Scripts Python modulares
│   ├── dataset.py          # Validação e geração do data.yaml
│   ├── evaluate.py         # Avaliação e geração de métricas
│   ├── analise_erros.py    # Análise de falsos positivos/negativos
│   └── demo.py             # Inferência em imagem/vídeo/webcam
│
├── notebooks/              # Notebooks Jupyter / Colab
│   └── train_colab.ipynb   # Notebook de treino (rodar no Colab com GPU T4)
│
├── results/                # Métricas e gráficos gerados pelos scripts
│   ├── metricas.json       # ← gerado por evaluate.py (nunca inventado)
│   ├── curva_pr.png
│   └── confusion_matrix.png
│
├── models/                 # Pesos treinados (.pt) — NÃO versionados no git
│   └── .gitkeep
│
├── data/                   # Dataset local — NÃO versionado no git
│   └── .gitkeep
│
├── examples/               # Imagens e vídeos de exemplo para demo
│   └── .gitkeep
│
├── docs/                   # Documentação do projeto
│   ├── relatorio.md        # Relatório técnico completo
│   ├── relatorio.pdf       # Versão exportada do relatório
│   └── guia_de_estudo.md   # Fichas de função + conceitos + 20 perguntas
│
├── .gitignore
├── requirements.txt
└── README.md               # Este arquivo
```

---

## Requisitos e Instalação

### Pré-requisitos
- Python 3.10+
- pip
- *(Opcional)* GPU NVIDIA com CUDA 11.8+ para inferência acelerada localmente

### Instalação

```bash
# 1. Clone o repositório (mantenha privado!)
git clone <URL_DO_SEU_REPO>
cd pothole-detection-gs

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
# ou
venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Execução Rápida

> ⚠️ **Os pesos do modelo treinado (`models/best.pt`) não estão no repositório.**
> Faça o treino no Colab (ver seção abaixo) e baixe o `best.pt` para `models/`.

### 1. Validar o dataset e gerar data.yaml

```bash
python src/dataset.py --data_dir data/ --output data/data.yaml
```

### 2. Avaliar no conjunto de teste

```bash
python src/evaluate.py --weights models/best.pt --data data/data.yaml
# Gera results/metricas.json e os gráficos
```

### 3. Analisar erros

```bash
python src/analise_erros.py --weights models/best.pt --data data/data.yaml
```

### 4. Demo

```bash
# Em imagem
python src/demo.py --source examples/imagem.jpg --weights models/best.pt

# Em vídeo
python src/demo.py --source examples/video.mp4 --weights models/best.pt

# Webcam (índice 0)
python src/demo.py --source 0 --weights models/best.pt
```

---

## Treino no Google Colab

1. Abra `notebooks/train_colab.ipynb` no [Google Colab](https://colab.research.google.com/)
2. Em **Runtime → Change runtime type**, selecione **GPU → T4**
3. Execute célula por célula — cada uma é autocontida e comentada
4. Ao final, baixe `runs/detect/train/weights/best.pt` e salve em `models/`

---

## Avaliação e Métricas

> 📊 Todas as métricas abaixo são geradas automaticamente por `src/evaluate.py`
> e armazenadas em `results/metricas.json`. **Nenhum número é inventado.**

| Métrica | Valor |
|---------|-------|
| Precisão (P) | *(gerado após treino)* |
| Recall (R) | *(gerado após treino)* |
| mAP@50 | *(gerado após treino)* |
| mAP@50-95 | *(gerado após treino)* |

---

## Demo

*(Screenshot ou GIF da demo serão adicionados após o treino)*

---

## Resultados Obtidos

*(Esta seção será preenchida automaticamente a partir de `results/metricas.json` após a execução dos scripts de avaliação)*

---

## Referências

- Jocher, G. et al. **Ultralytics YOLOv8** (2023). https://github.com/ultralytics/ultralytics
- Redmon, J. et al. **You Only Look Once: Unified, Real-Time Object Detection**. CVPR 2016.
- ONU. **Agenda 2030 — ODS 11: Cidades e Comunidades Sustentáveis**. https://odsbrasil.gov.br/objetivo/objetivo?n=11
- FIAP. **Global Solution 2026.2 — CitizenTech**. Enunciado da disciplina Cognitive Computing.
