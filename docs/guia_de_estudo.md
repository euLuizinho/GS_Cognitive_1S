# GS_Cognitive_1S — GUIA DE ESTUDO DEFINITIVO & MANUAL DE ARGUIÇÃO
## Detecção Inteligente de Buracos em Vias Urbanas com YOLOv8
**FIAP · Global Solution 2026.2 · Cognitive Computing & Computer Vision**  
**Projeto Individual:** GS_Cognitive_1S | **Docente:** Prof. Arnaldo Jr | **Tema:** CitizenTech (ODS 11 e 9)

---

## 📌 PARTE 0: MAPA E DEFESA DA ESTRUTURA DE PASTAS DO MEU PROJETO

Se o professor abrir o repositório na minha frente ou perguntar *"Como você organizou o seu projeto?"*, este é o mapa exato e a justificativa de engenharia de cada pasta:

```text
pothole-detection-gs/
│
├── 📄 .gitignore                ← Regras de exclusão do Git (bloqueia arquivos binários e temporários)
├── 📄 README.md                 ← Cartão de visitas com métricas reais, tabelas e instruções de execução
├── 📄 requirements.txt          ← Lista comentada das bibliotecas Python do ecossistema
│
├── 📁 src/                      ← O MOTOR: Scripts modulares em Python (uma função por responsabilidade)
│   ├── __init__.py              ← Torna a pasta um pacote Python importável
│   ├── dataset.py               ← Valida pares imagem/rótulo e gera o data.yaml oficial
│   ├── evaluate.py              ← Avalia o conjunto de teste e extrai o results/metricas.json
│   ├── analise_erros.py         ← Cataloga 10 Falsos Positivos e 10 Falsos Negativos
│   ├── demo.py                  ← APLICAÇÃO AO VIVO: inferência com HUD, contador e medidor de FPS
│   └── gerar_pdf.py             ← Compila o relatório acadêmico para PDF sem sobreposição
│
├── 📁 notebooks/                ← O LABORATÓRIO DE TREINAMENTO (Nuvem)
│   └── train_colab.ipynb        ← Notebook de 9 células executado na GPU T4 do Google Colab
│
├── 📁 results/                  ← AS EVIDÊNCIAS: Dados comprovados (NENHUM número inventado)
│   ├── metricas.json            ← JSON oficial das métricas (P: 91,96%, R: 85,37%, mAP: 89,02%)
│   ├── curvas_treinamento.png   ← Gráficos de perda (loss) e métricas pelas 50 épocas
│   ├── confusion_matrix.png     ← Matriz de confusão absoluta
│   ├── confusion_matrix_normalized.png ← Matriz de confusão normalizada em porcentagem
│   ├── amostra_predicoes_val.jpg← Lote de teste com caixas ciano e scores calculados
│   └── amostra_ground_truth_val.jpg ← Lote original com as anotações humanas reais
│
├── 📁 models/                   ← OS PESOS TREINADOS
│   └── best.pt                  ← O "cérebro" da rede: melhor modelo treinado (apenas 6,5 MB)
│
├── 📁 examples/                 ← MATERIAL DE DEMONSTRAÇÃO RÁPIDA NA BANCA
│   ├── buraco_rua.jpg           ← Foto individual de rua com buraco para o demo.py abrir na hora
│   └── exemplo_buracos.jpg      ← Amostra ampla de teste
│
├── 📁 docs/                     ← DOCUMENTAÇÃO ACADÊMICA E ESTUDO
│   ├── relatorio.md             ← O relatório técnico completo em texto Markdown
│   ├── relatorio.pdf            ← O PDF oficial diagramado de 4 páginas (entrega no portal)
│   └── guia_de_estudo.md        ← Meu manual de estudo com conceitos e as 20 perguntas da banca
│
└── 📁 data/                     ← DATASET LOCAL (fica no computador, excluído pelo .gitignore)
```

### Como defender cada pasta perante a banca:
- **`src/` (Desenvolvimento e Rigor Técnico — Vale 2,5 pts):** Em vez de fazer um código monolítico desorganizado, adotei o princípio da responsabilidade única (Single Responsibility Principle) e desacoplamento. Cada função possui uma docstring clara em português explicando o que faz, parâmetros e por que existe.
- **`notebooks/` (Reprodutibilidade):** Comprova que o treinamento foi executado de verdade célula por célula em hardware acelerado (GPU Tesla T4 no Colab), sem atalhos.
- **`results/` (Comprovação das Métricas — Vale 2,0 pts):** Cumpre a regra de ouro de nunca inventar números. Todos os dados do relatório saíram diretamente do `metricas.json` e dos gráficos oficiais de convergência gerados pelo Ultralytics.
- **`models/` (Pesos):** Guarda o arquivo `best.pt` (6,5 MB), que é o artefato final compilado que o `demo.py` carrega para inferir em tempo real.
- **`examples/` (Segurança da Apresentação):** Fornece imagens reais prontas para a banca testar na hora sem depender de internet.
- **`docs/` (Clareza de Documentação — Vale 1,0 pt):** Centraliza o relatório formal exigido no portal e este guia de estudos.

---

## 📌 PARTE 1: ROTEIRO PRÁTICO DA MINHA APRESENTAÇÃO EM SALA

A avaliação presencial vale **3,0 pontos na minha nota** (1,5 pts para a demonstração da aplicação funcional + 1,5 pts para as respostas da arguição individual). Aqui está o roteiro de como conduzir a apresentação:

### 1.1 Preparação no Computador do Laboratório (2 minutos)
1. Abro o navegador no GitHub privado do meu projeto.
2. Abro uma aba com o arquivo **`docs/relatorio.pdf`** aberto (para mostrar as tabelas e gráficos se o professor pedir).
3. Abro o terminal (PowerShell ou Prompt de Comando) na pasta do projeto:
   ```bash
   cd C:\Users\...\pothole-detection-gs
   ```

### 1.2 O Momento da Demonstração Prática (1,5 pts)
Executo o script de demonstração apontando para a imagem de teste que deixei pronta em `examples/`:
```bash
python src/demo.py --source examples/buraco_rua.jpg --weights models/best.pt --conf 0.15
```
*O que falar enquanto a janela abre na tela:*
> *"Professor, este é o meu projeto em execução em tempo real (GS_Cognitive_1S). Desenvolvi um detector de objetos baseado na arquitetura YOLOv8n (nano), voltado ao tema CitizenTech da Global Solution. Treinei o modelo individualmente no Google Colab com foco na identificação de buracos e falhas viárias. Na barra superior (HUD), o meu sistema faz a contagem instantânea de buracos identificados e calcula o FPS e a latência de inferência quadro a quadro. Como estou rodando aqui na CPU comum do laboratório, o tempo de inferência é de aproximadamente 35 a 45 ms, o que garante cerca de 24 quadros por segundo — plenamente suficiente para analisar o pavimento a cada meio metro com um veículo trafegando a 50 km/h."*

*(Opcional: Se o computador tiver webcam funcional, posso rodar `python src/demo.py --source 0 --weights models/best.pt` e apontar a câmera para uma foto de buraco na tela do meu celular para demonstrar o fluxo contínuo de vídeo).*

### 1.3 O Momento da Arguição Individual (1,5 pts)
O professor vai olhar o meu código, o PDF e fazer 2 a 3 perguntas conceituais. Falo com firmeza na primeira pessoa: *"eu escolhi"*, *"eu treinei"*, *"eu analisei"*.

---

## 📌 PARTE 2: CONCEITOS TEÓRICOS EXPLICADOS COM OS MEUS NÚMEROS REAIS

### 1. O que é IoU (Intersection over Union)?
- **Conceito:** É a métrica que mede a sobreposição geométrica entre a caixa que o modelo previu ($A$) e a caixa real anotada pelo humano ($B$).
  $$\text{IoU} = \frac{\text{Área da Interseção } (A \cap B)}{\text{Área da União } (A \cup B)}$$
- **No meu projeto:** Adotei o limiar clássico de $\text{IoU} \ge 0.50$. Se a caixa prevista cobrir pelo menos 50% da área do buraco real, considero como detecção correta.

### 2. O que é Precisão (Precision) e quanto deu no meu modelo?
- **Conceito:** De tudo o que o modelo previu como buraco, quanto realmente era buraco?
  $$\text{Precisão} = \frac{\text{Verdadeiros Positivos (TP)}}{\text{Verdadeiros Positivos (TP)} + \text{Falsos Positivos (FP)}}$$
- **Meu Número Real:** **91,96%**.
- **O que significa na prática:** O meu modelo é altamente seguro contra alarmes falsos. De cada 100 notificações geradas, 92 são buracos reais e apenas 8 são falhas (como sombras ou remendos).

### 3. O que é Recall (Sensibilidade) e quanto deu no meu modelo?
- **Conceito:** De todos os buracos que existiam na pista, quantos o modelo conseguiu detectar?
  $$\text{Recall} = \frac{\text{Verdadeiros Positivos (TP)}}{\text{Verdadeiros Positivos (TP)} + \text{Falsos Negativos (FN)}}$$
- **Meu Número Real:** **85,37%**.
- **O que significa na prática:** O sistema captura 85 de cada 100 buracos existentes. Os 15 restantes são quase sempre buracos rasos ou poças d'água barrentas niveladas.

### 4. O que é mAP@50 (Mean Average Precision)?
- **Conceito:** É a média da área sob a curva Precisão $\times$ Recall sob o limiar de $\text{IoU} = 0.50$. Representa o equilíbrio ótimo entre precisão e cobertura.
- **Meu Número Real:** **89,02%**. Um valor expressivo para visão computacional em ambiente urbano externo descontrolado.

### 5. O que é mAP@50-95?
- **Conceito:** É a média do mAP calculada em 10 degraus sucessivos de sobreposição ($\text{IoU} = 0.50, 0.55, 0.60, \dots, 0.95$). Avalia a precisão milimétrica da delimitação da caixa.
- **Meu Número Real:** **58,56%**. Consistente para um modelo nano de 3 milhões de parâmetros.

### 6. O que é a Matriz de Confusão?
- **Conceito:** Tabela cruzada que mostra acertos (diagonal principal) e erros entre classes e com o fundo (*background*).
- **No meu projeto:** Comprova que o meu modelo quase nunca confunde buracos (`Pothole`) com tampas de bueiro (`Sewer Cover`), mostrando que ele aprendeu a textura metálica dos bueiros.

---

## 📌 PARTE 3: FICHAS TÉCNICAS DAS FUNÇÕES QUE EU ESCREVI

### MÓDULO 1: `src/dataset.py`
1. **`listar_imagens(pasta)`**: Varre a pasta e retorna arquivos de imagem válidos (`.jpg`, `.png`, `.bmp`). *Por que criei:* Centraliza extensões aceitas.
2. **`listar_anotacoes(pasta)`**: Retorna arquivos `.txt` de anotação. *Por que criei:* Isola a leitura de rótulos.
3. **`verificar_par_imagem_rotulo(img_dir, lbl_dir)`**: Checa paridade entre foto e anotação. *Por que criei:* Evita dados órfãos que poluem o treino.
4. **`validar_formato_rotulo(arquivo_txt)`**: Valida se há 5 campos numéricos e limites em $[0.0, 1.0]$. *Por que criei:* Previne falhas no DataLoader.
5. **`detectar_classes(lbl_dir)`**: Identifica IDs numéricos únicos de classes. *Por que criei:* Monta o arquivo YAML correto.
6. **`validar_estrutura_dataset(data_dir)`**: Valida pastas `train`, `val`, `test`. *Por que criei:* Ponto único de auditoria prévia.
7. **`gerar_data_yaml(data_dir, classes, output, splits)`**: Escreve o arquivo `data.yaml`. *Por que criei:* Configuração automática do Ultralytics.

### MÓDULO 2: `src/evaluate.py`
8. **`carregar_modelo_yolo(caminho_pesos)`**: Instancia o modelo YOLO validando se o `.pt` existe. *Por que criei:* Tratamento robusto de erro.
9. **`verificar_split_disponivel(caminho_yaml, split)`**: Aplica fallback para `val` se `test` não constar no YAML. *Por que criei:* Flexibilidade entre datasets.
10. **`executar_avaliacao(modelo, caminho_yaml, ...)`**: Roda o pipeline nativo de `.val()` do Ultralytics. *Por que criei:* Cálculo padronizado de métricas.
11. **`extrair_dicionario_metricas(resultados_val, split)`**: Extrai precisão, recall, mAP e tempos de latência. *Por que criei:* Desacopla da biblioteca.
12. **`salvar_metricas_json(metricas, caminho)`**: Salva o arquivo `results/metricas.json`. *Por que criei:* Comprovação auditável sem números manuais.
13. **`copiar_artefatos_graficos(origem, destino)`**: Transfere gráficos para `results/`. *Por que criei:* Mantém relatórios integrados.

### MÓDULO 3: `src/analise_erros.py`
14. **`carregar_rotulos_ground_truth(txt, larg, alt)`**: Desnormaliza $[cx, cy, w, h]$ para pixels absolutos. *Por que criei:* Permite cálculo exato de IoU.
15. **`calcular_iou(box_a, box_b)`**: Calcula a sobreposição espacial. *Por que criei:* Base matemática da classificação de acertos e erros.
16. **`categorizar_predicoes_imagem(gt, pred, iou)`**: Classifica em TP, FP e FN. *Por que criei:* Isola e conta as falhas do modelo.
17. **`desenhar_anotacoes_erro(img, gt, fp, fn)`**: Desenha Verde=GT, Vermelho=FP, Laranja=FN. *Por que criei:* Visualização autoexplicativa.
18. **`mapear_padroes_visuais_recorrentes()`**: Mapeia causas físicas de falhas (sombras, remendos). *Por que criei:* Embasamento qualitativo para banca.

### MÓDULO 4: `src/demo.py`
19. **`carregar_detector_yolo(pesos, device)`**: Inicializa modelo em CUDA ou CPU. *Por que criei:* Portabilidade para o PC do laboratório.
20. **`determinar_tipo_fonte(fonte_str)`**: Reconhece se é webcam (`0`), vídeo (`.mp4`) ou foto (`.jpg`). *Por que criei:* Versatilidade de apresentação.
21. **`executar_inferencia_frame(modelo, frame, ...)`**: Roda o forward pass e cronometra milissegundos. *Por que criei:* Medição precisa de latência.
22. **`desenhar_deteccoes_frame(frame, deteccoes)`**: Desenha bboxes ciano e rótulos legíveis. *Por que criei:* Contraste nítido para telão.
23. **`desenhar_hud_informacoes(frame, buracos, fps, latencia)`**: Renderiza o painel superior. *Por que criei:* Exigência explícita do enunciado da matéria.
24. **`processar_modo_imagem()` / `processar_modo_stream()`**: Gerenciam loops e teclas `q` (sair) e `s` (screenshot). *Por que criei:* Separação limpa de fluxo.

---

## 📌 PARTE 4: AS 20 PERGUNTAS MAIS PROVÁVEIS DA BANCA (RESPOSTAS INDIVIDUAIS)

### Problema e Alinhamento CitizenTech
**1. Por que este projeto é classificado como CitizenTech e não apenas uma solução governamental tradicional?**  
*Resposta:* "Porque desenvolvi a solução a partir da necessidade real do cidadão comum e para rodar em dispositivos acessíveis (smartphones e câmeras veiculares de baixo custo), democratizando o monitoramento do espaço público. Em vez de depender de burocracias ou ouvidorias lentas, qualquer cidadão ou motorista de transporte público atua como agente colaborativo da zeladoria urbana."

**2. Como a sua solução se conecta especificamente aos ODS 11 e 9 da ONU?**  
*Resposta:* "No ODS 11 (Cidades Sustentáveis, Meta 11.2), a detecção precoce de buracos reduz diretamente o risco de acidentes graves com motociclistas, ciclistas e pedestres. No ODS 9 (Infraestrutura, Meta 9.1), permite à gestão pública realizar manutenção preventiva baseada em dados, otimizando o gasto tributário e tornando o pavimento mais resiliente."

**3. Qual é o diferencial prático de usar Visão Computacional em vez de acelerômetros em smartphones?**  
*Resposta:* "Acelerômetros só registram o impacto mecânico *depois* que a roda já caiu no buraco, gerando desconforto e risco de acidente imediato. A Visão Computacional atua de forma preditiva e ótica: enxerga a descontinuidade a metros de distância antes que o veículo passe por ela, além de fornecer o registro visual objetivo do dano."

### Dados e Dataset
**4. Qual dataset você utilizou, qual a licença e quantas imagens ele possui?**  
*Resposta:* "Utilizei o Pothole Detection Dataset do Intel Unnati Training Program, hospedado no Roboflow Universe sob licença MIT (permissiva para pesquisa acadêmica). Ele contém mais de 2.700 imagens veiculares reais. O meu conjunto de validação independente utilizou 491 imagens contendo 1.079 anotações."

**5. Por que foi importante o dataset ter classes separadas para 'Sewer Cover' e 'Drain Hole' além de 'Pothole'?**  
*Resposta:* "Para evitar que o meu modelo sofresse de falsos alarmes crônicos. Em vias urbanas, tampas de bueiro metálicas e ralos pluviais têm desníveis e sombras circulares parecidas com buracos. Se eu rotulasse tudo como uma classe só, a prefeitura receberia alertas de manutenção onde existe apenas um bueiro regular da companhia de saneamento."

**6. Como você garantiu que não houve vazamento de dados (data leakage)?**  
*Resposta:* "Mantive a separação estrita dos dados em splits isolados de treino e validação desde o download, e o meu script `src/dataset.py` validou a integridade e paridade dos arquivos sem compartilhamento de imagens entre os grupos."

### Arquitetura e Modelagem
**7. Por que você escolheu o YOLOv8n (nano) em vez de versões maiores como YOLOv8m ou YOLOv8x?**  
*Resposta:* "Por um compromisso de engenharia voltado a Edge Computing. O YOLOv8n possui apenas 3,13 milhões de parâmetros e 8,6 GFLOPs (~6,5 MB no disco). Uma versão maior consumiria muita memória e bateria, inviabilizando a execução em tempo real em smartphones ou CPUs sem placa de vídeo dedicada."

**8. O que é Transfer Learning e como ele foi aplicado no seu treinamento?**  
*Resposta:* "É a técnica de iniciar a rede neural a partir de pesos previamente treinados em uma grande base de dados (o Microsoft COCO com 80 classes universais), em vez de começar com pesos aleatórios. Assim, as primeiras camadas convolucionais já entraram sabendo reconhecer bordas, curvas e gradientes de luz, necessitando apenas de fine-tuning para aprender a geometria de buracos no asfalto."

**9. O que significa o YOLO ser uma arquitetura 'Single-Shot'?**  
*Resposta:* "Significa que ele prevê as caixas delimitadoras e as probabilidades de classe em uma única passada (*forward pass*) da imagem pela rede neural convolucional. Diferente de redes de dois estágios (como Faster R-CNN), ele não gera regiões candidatas separadamente, o que o torna de 5 a 10 vezes mais rápido."

**10. Quais funções de perda (losses) compõem a função de custo do YOLOv8?**  
*Resposta:* "Uma perda combinada com três termos: CIoU Loss para penalizar erros de proporção e localização espacial da caixa; Binary Cross-Entropy (BCE) para a classificação das categorias; e Distribution Focal Loss (DFL) para refinar com precisão os contornos e limites irregulares das bordas."

### Métricas e Resultados
**11. Quais foram as métricas finais obtidas no seu modelo?**  
*Resposta:* "No conjunto de validação com 491 imagens e 1.079 instâncias, obtive 91,96% de Precisão, 85,37% de Recall, 89,02% de mAP@50 e 58,56% de mAP@50-95, com inferência média de 4,8 ms em GPU e cerca de 42 ms em CPU local."

**12. Explique a diferença entre Precisão e Recall com base nos resultados do seu projeto.**  
*Resposta:* "A minha Precisão de 91,96% mede a confiabilidade dos alarmes: quando o modelo diz que viu um buraco, ele está certo em quase 92% das vezes, evitando despachar equipes de tapa-buraco à toa. O Recall de 85,37% mede a cobertura: de cada 100 buracos existentes na via, o sistema consegue identificar 85 deles."

**13. Por que o Recall (85,37%) deu menor que a Precisão (91,96%)?**  
*Resposta:* "Porque ajustei o limiar de confiança para 0.25 para priorizar predições seguras. Além disso, buracos muito rasos (menos de 2 a 3 cm) ou preenchidos com água barrenta pós-chuva perdem o degrau de sombra que indica profundidade, fazendo com que a rede opte por não disparar a detecção para evitar alarme falso."

**14. O que é o mAP@50 e por que ele é a métrica principal em detecção de objetos?**  
*Resposta:* "O mAP@50 é a Média da Precisão Média calculada sob um limiar de sobreposição IoU de 50%. Ele sintetiza em um único número a qualidade da curva Precisão $\times$ Recall em todos os níveis de confiança. O meu resultado de 89,02% comprova alta estabilidade do detector."

### Análise Crítica e Diagnóstico de Erros
**15. Quais são os principais casos em que o seu modelo gera Falsos Positivos (alarmes falsos)?**  
*Resposta:* "Principalmente sombras recortadas de copas de árvores em dias de sol a pino, pois a sombra irregular sobre o asfalto cinza claro cria um gradiente de borda que simula a cavidade escura de um buraco. Outro caso são remendos recentes de asfalto novo com piche muito preto em contraste com pistas velhas."

**16. Quais são os principais casos em que o seu modelo gera Falsos Negativos (omissões)?**  
*Resposta:* "Buracos inundados por água de chuva ou lama, porque a lâmina d'água atua como espelho ou nivela a superfície com a pista, eliminando a percepção monocular de profundidade; e desgastes superficiais rasos que não possuem relevo geométrico expressivo."

**17. Como esses erros poderiam ser mitigados em uma versão 2.0 da sua solução?**  
*Resposta:* "Através de rastreamento temporal (*tracking*) entre quadros consecutivos (uma sombra se move diferente de um buraco fixo na via), fusão sensorial com acelerômetros do veículo e expansão do dataset com exemplos rotulados de chuva e pavimento molhado."

### Código e Engenharia de Software
**18. Como o seu código está estruturado para garantir manutenção e clareza?**  
*Resposta:* "Segui o princípio da responsabilidade única (Single Responsibility Principle): cada função tem uma docstring detalhada em português explicando o que faz, parâmetros e o porquê de existir. Desacoplei o pipeline em quatro módulos: `dataset.py` (validação e YAML), `evaluate.py` (métricas oficiais e JSON), `analise_erros.py` (diagnóstico qualitativo de FPs/FNs) e `demo.py` (visualização em tempo real)."

**19. Como funciona o cálculo de FPS e contagem de buracos em tempo real no `demo.py`?**  
*Resposta:* "A inferência roda quadro a quadro e cronometro o tempo com `time.perf_counter()`. Para o FPS não oscilar na tela, apliquei um filtro de suavização exponencial ($FPS = 0.9 \cdot FPS_{\text{ant}} + 0.1 \cdot FPS_{\text{inst}}$). A contagem é a soma dos bounding boxes com confiança acima de 0.25 desenhados no cabeçalho translúcido (HUD)."

**20. Como seria a implantação desta solução em larga escala em uma cidade como São Paulo?**  
*Resposta:* "Adotando arquitetura de Edge AI: a inferência roda localmente no dispositivo embarcado do veículo. Em vez de enviar vídeos pesados que sobrecarregariam redes móveis, o dispositivo transmite apenas pequenas mensagens JSON (<1 KB) com coordenadas GPS, timestamp e severidade para uma API em nuvem, que agrupa detecções repetidas em um raio de 5 metros e plota mapas de calor para a rota das equipes de manutenção."

---

## 📌 PARTE 5: CHECKLIST FINAL ANTES DE ENTRAR NA SALA

- [ ] Repositório clonado e funcionando localmente.
- [ ] `docs/relatorio.pdf` salvo no computador e acessível no GitHub.
- [ ] Teste de comando executado previamente: `python src/demo.py --source examples/buraco_rua.jpg --weights models/best.pt --conf 0.15`.
- [ ] Leitura prévia deste guia de estudo para fixar os meus números: **Precisão: 91,96%**, **Recall: 85,37%**, **mAP@50: 89,02%** e **FPS: ~24 a 111**.
