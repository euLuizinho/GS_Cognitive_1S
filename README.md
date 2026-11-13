# Detecção Inteligente de Buracos em Vias Urbanas com YOLOv8
### Global Solution 2026.2 — FIAP | Cognitive Computing | CitizenTech

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![ODS 11 & 9](https://img.shields.io/badge/ONU-ODS%2011%20%7C%20ODS%209-green.svg)](https://brasil.un.org/pt-br/sdgs)
[![Licença MIT](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
Luiz Fabiano Nascimento Vale da Silva RM553529

---

## Sumário
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Alinhamento CitizenTech e ODS](#alinhamento-citizentech-e-ods)
3. [Resultados e Métricas Reais](#resultados-e-métricas-reais)
4. [Estrutura do Repositório](#estrutura-do-repositório)
5. [Instalação e Ambiente](#instalação-e-ambiente)
6. [Guia de Execução](#guia-de-execução)
   - [Validação do Dataset](#1-validação-do-dataset)
   - [Avaliação Oficial](#2-avaliação-oficial-de-métricas)
   - [Diagnóstico de Erros](#3-diagnóstico-de-erros-falsos-positivosnegativos)
   - [Demonstração em Tempo Real](#4-demonstração-em-tempo-real-demo)
7. [Treinamento no Google Colab](#treinamento-no-google-colab)
8. [Análise Crítica e Limitações](#análise-crítica-e-limitações)
9. [Referências](#referências)

---

## Visão Geral do Projeto

A deterioração do pavimento asfáltico em vias urbanas é um dos principais fatores geradores de acidentes de trânsito, danos patrimoniais e lentidão na mobilidade coletiva nas cidades brasileiras. O modelo tradicional de manutenção municipal opera de maneira reativa, dependendo de fiscalizações presenciais escassas ou de chamados manuais abertos por cidadãos em ouvidorias com longos tempos de resposta.

Este projeto desenvolve uma solução de **Visão Computacional e Inteligência Artificial** baseada na arquitetura **YOLOv8n** (*nano*), capaz de identificar, catalogar e contar buracos na via em tempo real com **91,96% de precisão**. O sistema atua como motor de inferência embarcável em smartphones e veículos de serviços urbanos (ônibus, viaturas de coleta de lixo, táxis), transformando qualquer rota em uma varredura automática de zeladoria colaborativa.

---

## Alinhamento CitizenTech e ODS

No escopo da Global Solution 2026.2 (**CitizenTech: Da Cidade ao Campo, Tecnologia para a Vida Cotidiana**), a solução democratiza o monitoramento do espaço público urbano:

| ODS da ONU | Meta Específica | Contribuição Concreta do Projeto |
|---|---|---|
| **ODS 11: Cidades e Comunidades Sustentáveis** | **Meta 11.2:** Proporcionar o acesso a sistemas de transporte seguros, acessíveis e sustentáveis para todos. | Reduz riscos de acidentes graves para ciclistas, motociclistas e pedestres através da detecção precoce de irregularidades viárias. |
| **ODS 9: Indústria, Inovação e Infraestrutura** | **Meta 9.1:** Desenvolver infraestrutura de qualidade, confiável, sustentável e resiliente. | Fornece auditoria baseada em dados visuais objetivos, otimizando o orçamento público de recapeamento e tapa-buracos. |

---

## Resultados e Métricas Reais

> ⚠️ **Princípio de Rigor Técnico:** Nenhuma métrica abaixo foi estimada ou digitada manualmente. Todos os valores foram extraídos diretamente do artefato oficial [`results/metricas.json`](results/metricas.json), gerado após 50 épocas de treinamento do modelo `YOLOv8n` com GPU NVIDIA Tesla T4.

### Desempenho Global no Conjunto de Validação (491 imagens, 1.079 instâncias)

| Métrica Oficial | Valor Obtido | Significado Prático |
|---|---|---|
| **Precisão (Precision - P)** | **91,96%** | Quando o modelo aponta um buraco, ele acerta em 92 de cada 100 detecções (quase isento de alarmes falsos). |
| **Recall (Sensibilidade - R)** | **85,37%** | O sistema captura 85 de cada 100 irregularidades reais presentes na pista. |
| **mAP @ 0.50** | **89,02%** | Média da Precisão Média calculada sob limiar de sobreposição $\text{IoU} \ge 0.50$. |
| **mAP @ 0.50:0.95** | **58,56%** | Métrica estrita ponderada em múltiplos degraus de sobreposição espacial. |

### Desempenho por Categoria

O modelo foi capacitado a diferenciar três elementos distintos da infraestrutura viária:

| Categoria | Instâncias de Validação | Precisão ($P$) | Recall ($R$) | mAP@50 | mAP@50:95 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Pothole** (Buraco na pista) | 1.025 | **89,70%** | **78,20%** | **86,80%** | 60,60% |
| **Drain Hole** (Ralo / Boca de lobo) | 14 | **97,00%** | **92,90%** | **92,50%** | 60,80% |
| **Sewer Cover** (Tampa de esgoto) | 40 | **89,20%** | **85,00%** | **87,80%** | 54,30% |

### Latência e Eficiência Computacional

- **Tamanho do modelo:** 73 camadas, 3.133.193 parâmetros, 8,6 GFLOPs (~6,5 MB no disco).
- **Tempo por imagem (GPU Tesla T4):** 1,8 ms (pré) + 4,8 ms (inferência) + 2,4 ms (pós) = **9,0 ms** (~**111 FPS**).
- **Tempo por imagem (CPU Intel i5/i7):** ~35 a 55 ms (~**18 a 28 FPS**), plenamente viável para operação em tempo real sem acelerador gráfico.

---

## Estrutura do Repositório

```text
GS_Cognitive_1S/
├── .gitignore               # Exclui datasets brutos, pesos binários e temporários
├── README.md                # Este documento com comprovação formal
├── requirements.txt         # Dependências comentadas do ecossistema Python
│
├── src/                     # Módulos Python modulares (uma responsabilidade por função)
│   ├── __init__.py
│   ├── dataset.py           # Validação estrutural de anotações e geração do data.yaml
│   ├── evaluate.py          # Avaliação formal e consolidação de results/metricas.json
│   ├── analise_erros.py     # Diagnóstico de FPs/FNs e mapeamento taxonômico
│   └── demo.py              # Aplicação de demonstração ao vivo (webcam, vídeo, imagem)
│
├── notebooks/               # Treinamento reproduzível em nuvem
│   └── train_colab.ipynb    # Notebook com 9 células documentadas para execução em GPU T4
│
├── results/                 # Artefatos comprovados de avaliação
│   ├── metricas.json        # Arquivo de dados brutos das métricas
│   ├── confusion_matrix.png # Matriz de confusão absoluta
│   ├── confusion_matrix_normalized.png # Matriz de confusão normalizada
│   ├── curvas_treinamento.png # Curvas de loss e evolução das métricas por época
│   └── amostra_predicoes_val.jpg # Batch oficial de validação com caixas e scores
│
├── models/                  # Diretório local para pesos treinados (.pt)
│   └── best.pt              # Modelo ótimo treinado (6,5 MB)
│
├── examples/                # Imagens e amostras para demonstração rápida em banca
│   └── exemplo_buracos.jpg
│
└── docs/                    # Relatório acadêmico e guia da arguição
    ├── relatorio.md         # Relatório técnico completo formatado
    ├── relatorio.pdf        # Documento exportado para entrega no portal
    └── guia_de_estudo.md    # Fichas de função, conceitos e 20 perguntas da banca
```

---

## Instalação e Ambiente

Recomenda-se o uso de **Python 3.10 ou superior**:

```bash
# 1. Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd GS_Cognitive_1S

# 2. Criar e ativar ambiente virtual
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt
```

---

## Guia de Execução

### 1. Validação do Dataset
Verifica a integridade das imagens, pares de rótulos `.txt` e faixas numéricas de coordenadas:
```bash
python src/dataset.py --data_dir data/ --output data/data.yaml
```

### 2. Avaliação Oficial de Métricas
Roda o pipeline do Ultralytics no conjunto de dados, atualizando [`results/metricas.json`](results/metricas.json) e gerando gráficos:
```bash
python src/evaluate.py --weights models/best.pt --data data/data.yaml --split val
```

### 3. Diagnóstico de Erros (Falsos Positivos/Negativos)
Compara predições com Ground Truth via IoU e extrai as amostras de erro:
```bash
python src/analise_erros.py --weights models/best.pt --data data/data.yaml --max_amostras 10
```

### 4. Demonstração em Tempo Real (Demo)
Para a apresentação ao vivo na banca, utilize qualquer uma das três modalidades:

```bash
# Modo 1: Demonstração com imagem estática (ótimo para computadores sem webcam)
python src/demo.py --source examples/exemplo_buracos.jpg --weights models/best.pt

# Modo 2: Demonstração com vídeo gravado (.mp4)
python src/demo.py --source examples/video_rua.mp4 --weights models/best.pt

# Modo 3: Demonstração ao vivo via webcam conectada (câmera índice 0)
python src/demo.py --source 0 --weights models/best.pt
```
*Controles durante a demonstração:*
- Pressione **`q`** para encerrar a visualização.
- Pressione **`s`** para salvar uma screenshot instantânea em `results/`.

---

## Treinamento no Google Colab

O processo de treinamento foi projetado para execução no ambiente gratuito do **Google Colab com GPU NVIDIA Tesla T4**:
1. Abra [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) no Google Colab.
2. Defina o runtime: **Ambiente de Execução → Alterar tipo → T4 GPU**.
3. Execute sequencialmente as 9 células:
   - Célula 1 a 3: Verificação de hardware, instalação e download do dataset via Roboflow Universe.
   - Célula 4 e 5: Inspeção estatística e visualização prévia das caixas de treino.
   - Célula 6: Treinamento de 50 épocas do `YOLOv8n` com otimizador automático e parada antecipada (*early stopping*).
   - Célula 7 e 8: Avaliação quantitativa e amostragem visual de predições.
   - Célula 9: Empacotamento automático de `best.pt` e gráficos no arquivo `artefatos_gs2026.zip`.

---

## Análise Crítica e Limitações

A avaliação qualitativa conduzida por `src/analise_erros.py` identificou os padrões de borda do modelo:

1. **Falsos Positivos:**
   - **Sombras Arbóreas Densas:** A incidência de sol perpendicular cruzando copas de árvores projeta silhuetas escuras irregulares no asfalto claro, gerando bordas de alto gradiente que mimetizam a depressão de um buraco.
   - **Remendos Recentes de Asfalto:** Aplicações de piche novo (asfalto escuro fosco) contrastando com pistas desgastadas (cinza claro) ativam esporadicamente a detecção de cavidade.
2. **Falsos Negativos:**
   - **Buracos Inundados:** Lâminas de água barrenta nivelam visualmente a cavidade com a superfície da via, eliminando a sombra interior indicativa de profundidade.
   - **Erosão Inicial / Pavimento Esfarelado:** Desgastes superficiais inferiores a 3 cm de profundidade não geram descontinuidade textural suficiente para a resolução de entrada de $640 \times 640$ pixels.

---

## Referências

- **Jocher, G., Chaurasia, A., & Qiu, J.** (2023). *Ultralytics YOLOv8* (Version 8.0.0). GitHub: https://github.com/ultralytics/ultralytics.
- **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.** (2016). *You Only Look Once: Unified, Real-Time Object Detection*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- **Organização das Nações Unidas (ONU).** *Objetivo de Desenvolvimento Sustentável 11: Cidades e Comunidades Sustentáveis*. Agenda 2030.
- **FIAP.** *Enunciado Oficial da Global Solution 2026.2 — CitizenTech: Da Cidade ao Campo, Tecnologia para a Vida Cotidiana*. Disciplina Cognitive Computing, Computer Vision and IoT Systems (Prof. Arnaldo Jr).

