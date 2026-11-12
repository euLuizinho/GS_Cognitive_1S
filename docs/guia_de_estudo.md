# GUIA DE ESTUDO DEFINITIVO & MANUAL DE ARGUIÇÃO INDIVIDUAL
## Detecção Inteligente de Buracos em Vias Urbanas com YOLOv8
**FIAP · Global Solution 2026.2 · Cognitive Computing & Computer Vision**  
**Docente:** Prof. Arnaldo Jr | **Tema:** CitizenTech (ODS 11 e 9)

---

## 📌 PARTE 1: ROTEIRO PRÁTICO DA APRESENTAÇÃO EM SALA

A avaliação presencial vale **3,0 pontos na sua média** (1,5 pts para a demonstração da aplicação funcional + 1,5 pts para as respostas da arguição individual). Aqui está o roteiro de como conduzir a apresentação com segurança:

### 1.1 Preparação no Computador do Laboratório (2 minutos)
1. Abra o navegador no GitHub privado do seu projeto.
2. Abra uma aba com o arquivo **`docs/relatorio.pdf`** aberto (para mostrar as tabelas e gráficos se o professor pedir).
3. Abra o terminal (PowerShell ou Prompt de Comando) na pasta do projeto:
   ```bash
   cd C:\Users\...\pothole-detection-gs
   ```

### 1.2 O Momento da Demonstração Prática (1,5 pts)
Execute o script de demonstração apontando para a imagem de teste que deixamos pronta em `examples/`:
```bash
python src/demo.py --source examples/exemplo_buracos.jpg --weights models/best.pt
```
*O que falar enquanto a janela abre na tela:*
> *"Professor, esta é a nossa aplicação em tempo real. O modelo foi treinado por nós especificamente para a infraestrutura viária e opera em arquitetura Single-Shot (YOLOv8n). Na barra superior (HUD), o sistema faz a contagem instantânea de buracos identificados e calcula o FPS e a latência de inferência quadro a quadro. Como estamos rodando na CPU comum do laboratório, o tempo de inferência é de aproximadamente 35 a 45 ms, o que garante cerca de 24 quadros por segundo — plenamente suficiente para analisar o pavimento a cada meio metro com um veículo a 50 km/h."*

*(Opcional: Se o computador tiver webcam funcional, você pode rodar `python src/demo.py --source 0 --weights models/best.pt` e apontar a câmera para uma foto de buraco na tela do seu celular para demonstrar o fluxo de vídeo contínuo).*

### 1.3 O Momento da Arguição Individual (1,5 pts)
O professor vai olhar o seu código, o PDF e fazer 2 a 3 perguntas conceituais ou sobre o código. Mantenha a calma, fale com clareza e use os termos técnicos que estão detalhados nas partes a seguir deste guia.

---

## 📌 PARTE 2: CONCEITOS TEÓRICOS EXPLICADOS COM OS SEUS NÚMEROS REAIS

### 1. O que é IoU (Intersection over Union)?
- **Conceito:** É a métrica que mede a sobreposição geométrica entre a caixa que o modelo tentou adivinhar ($A$) e a caixa real anotada pelo humano ($B$).
  $$\text{IoU} = \frac{\text{Área da Interseção } (A \cap B)}{\text{Área da União } (A \cup B)}$$
- **No seu projeto:** Adotamos o limiar clássico de $\text{IoU} \ge 0.50$. Se a caixa do modelo cobrir pelo menos metade da área do buraco real, consideramos que ele acertou a localização.

### 2. O que é Precisão (Precision) e quanto deu no seu projeto?
- **Conceito:** De tudo o que o modelo disse que era buraco, quanto realmente era buraco?
  $$\text{Precisão} = \frac{\text{Verdadeiros Positivos (TP)}}{\text{Verdadeiros Positivos (TP)} + \text{Falsos Positivos (FP)}}$$
- **Seu Número Real:** **91,96%**.
- **O que significa na prática:** Seu modelo é extremamente seguro contra alarmes falsos. De cada 100 notificações geradas, 92 são buracos reais e apenas 8 são falhas (como sombras ou remendos).

### 3. O que é Recall (Sensibilidade) e quanto deu no seu projeto?
- **Conceito:** De todos os buracos que existiam na rua, quantos o modelo conseguiu enxergar?
  $$\text{Recall} = \frac{\text{Verdadeiros Positivos (TP)}}{\text{Verdadeiros Positivos (TP)} + \text{Falsos Negativos (FN)}}$$
- **Seu Número Real:** **85,37%**.
- **O que significa na prática:** O sistema tem alta cobertura, capturando 85 de cada 100 buracos existentes no asfalto. Os 15 que escapam são quase sempre buracos rasos ou poças d'água niveladas.

### 4. O que é mAP@50 (Mean Average Precision)?
- **Conceito:** É a área sob a curva Precisão $\times$ Recall calculada sob o limiar de $\text{IoU} = 0.50$, calculada para cada classe e depois feita a média geral. Representa o equilíbrio perfeito entre não dar alarme falso e não deixar passar nada.
- **Seu Número Real:** **89,02%**. Um valor acima de 80% em visão computacional externa (com iluminação descontrolada) é considerado de nível industrial/produção.

### 5. O que é mAP@50-95?
- **Conceito:** É a média do mAP calculada em 10 degraus sucessivos de sobreposição ($\text{IoU} = 0.50, 0.55, 0.60, \dots, 0.95$). Avalia se o modelo acerta o contorno milimétrico da caixa.
- **Seu Número Real:** **58,56%**. É um valor muito consistente para um modelo nano de apenas 3 milhões de parâmetros.

### 6. O que é a Matriz de Confusão?
- **Conceito:** Tabela cruzada que mostra onde a rede acertou (diagonal principal) e onde ela errou (linhas vs colunas de classes diferentes ou com o fundo/background).
- **No seu projeto:** Mostra que o modelo quase nunca confunde buracos (`Pothole`) com tampas de esgoto (`Sewer Cover`), comprovando que ele aprendeu a textura metálica dos bueiros.

---

## 📌 PARTE 3: FICHAS TÉCNICAS DAS FUNÇÕES DO CÓDIGO (UMA POR UMA)

### MÓDULO 1: `src/dataset.py`

#### 1. `listar_imagens(pasta: Path) -> list[Path]`
- **O que faz:** Varre a pasta recursivamente e retorna todos os arquivos com extensões de imagem válidas (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- **Por que existe:** Centraliza as extensões aceitas pelo OpenCV/Ultralytics, evitando duplicação de filtros no código.

#### 2. `listar_anotacoes(pasta: Path) -> list[Path]`
- **O que faz:** Retorna a lista de todos os arquivos de rótulo `.txt` no diretório.
- **Por que existe:** Isola a leitura de anotações da busca de imagens para permitir validação desacoplada.

#### 3. `verificar_par_imagem_rotulo(img_dir: Path, lbl_dir: Path) -> tuple[int, list[str]]`
- **O que faz:** Confere se para cada imagem existe um `.txt` com o mesmo nome na pasta de labels.
- **Por que existe:** Evita erros silenciosos de treino; imagens órfãs sem rótulo são tratadas como fundo e distorcem o cálculo da perda.

#### 4. `validar_formato_rotulo(arquivo_txt: Path) -> list[str]`
- **O que faz:** Lê linha a linha de cada anotação e valida se há exatamente 5 campos numéricos e se as coordenadas normalizadas estão em $[0.0, 1.0]$.
- **Por que existe:** Coordenadas fora de $[0, 1]$ ou textos corrompidos travam o DataLoader do PyTorch no meio do treino.

#### 5. `detectar_classes(lbl_dir: Path) -> list[int]`
- **O que faz:** Extrai os identificadores numéricos únicos presentes na primeira coluna dos rótulos.
- **Por que existe:** Garante que o arquivo `data.yaml` seja configurado exatamente com as classes reais que o dataset contém.

#### 6. `validar_estrutura_dataset(data_dir: Path) -> dict`
- **O que faz:** Orquestra a validação completa dos splits (`train`, `val`/`valid`, `test`) e consolida contagens e erros em um dicionário.
- **Por que existe:** Atua como ponto central de auditoria estrutural antes de subir dados para a nuvem.

#### 7. `gerar_data_yaml(data_dir, classes, output, splits)`
- **O que faz:** Escreve em disco o arquivo `data.yaml` com caminhos absolutos e nomes das classes no formato padrão Ultralytics.
- **Por que existe:** Automatiza o contrato entre os dados e o YOLO, eliminando erros manuais de digitação de paths.

---

### MÓDULO 2: `src/evaluate.py`

#### 8. `carregar_modelo_yolo(caminho_pesos: Path) -> YOLO`
- **O que faz:** Instancia o modelo YOLO a partir de um arquivo `.pt`, checando previamente se o arquivo existe em disco.
- **Por que existe:** Falha rápido (*fail-fast*) com mensagem amigável caso os pesos não tenham sido baixados.

#### 9. `verificar_split_disponivel(caminho_yaml: Path, split_desejado: str) -> str`
- **O que faz:** Verifica se o split `test` existe no YAML; se não existir, adota o split `val` como alternativa segura.
- **Por que existe:** Previne exceções em tempo de execução quando datasets usam nomes alternativos de partição.

#### 10. `executar_avaliacao(modelo, caminho_yaml, split, imgsz, conf, iou) -> object`
- **O que faz:** Executa o método nativo `.val()` da biblioteca Ultralytics no conjunto selecionado.
- **Por que existe:** Padroniza o cálculo das métricas oficiais de detecção (P, R, mAP) com NMS formal.

#### 11. `extrair_dicionario_metricas(resultados_val, split_utilizado) -> dict`
- **O que faz:** Converte o objeto interno do Ultralytics em um dicionário Python limpo com 4 casas decimais e tempos de latência.
- **Por que existe:** Desacopla a estrutura interna de bibliotecas terceiras do restante da nossa aplicação.

#### 12. `salvar_metricas_json(metricas: dict, caminho_arquivo: Path) -> None`
- **O que faz:** Serializa o dicionário de métricas em formato JSON indentado.
- **Por que existe:** Garante a comprovação estrita das métricas sem digitação manual, alimentando o relatório e o README.

#### 13. `copiar_artefatos_graficos(diretorio_origem, diretorio_destino) -> list[str]`
- **O que faz:** Copia os gráficos de matriz de confusão e curvas PxR da pasta de runs para a pasta `results/`.
- **Por que existe:** Mantém o repositório organizado e os artefatos prontos para compilação no relatório PDF.

---

### MÓDULO 3: `src/analise_erros.py`

#### 14. `carregar_rotulos_ground_truth(caminho_txt, largura_img, altura_img) -> list[dict]`
- **O que faz:** Desnormaliza as coordenadas relativas do YOLO para pixels absolutos $[x_1, y_1, x_2, y_2]$.
- **Por que existe:** Necessário para permitir o cálculo exato de sobreposição de IoU e desenho com OpenCV.

#### 15. `calcular_iou(box_a: list, box_b: list) -> float`
- **O que faz:** Calcula a razão entre a área de interseção e a área de união entre dois retângulos.
- **Por que existe:** Função fundamental para definir matematicamente se uma detecção acertou ou errou o alvo.

#### 16. `categorizar_predicoes_imagem(caixas_gt, predicoes, iou_limiar) -> tuple`
- **O que faz:** Compara predições com Ground Truths e classifica em Verdadeiros Positivos (TP), Falsos Positivos (FP) e Falsos Negativos (FN).
- **Por que existe:** Constrói a base analítica para entender exatamente onde o modelo está errando.

#### 17. `desenhar_anotacoes_erro(imagem_bgr, caixas_gt, caixas_fp, caixas_fn) -> np.ndarray`
- **O que faz:** Pinta as caixas na imagem com cores semânticas: Verde (GT real), Vermelho (FP) e Laranja (FN).
- **Por que existe:** Torna a visualização dos erros autoexplicativa para apresentação visual em banca.

#### 18. `mapear_padroes_visuais_recorrentes() -> dict`
- **O que faz:** Retorna a categorização técnica das causas físicas de erros (sombras, remendos, reflexos em poças).
- **Por que existe:** Fornece embasamento científico e qualitativo para a arguição técnica.

---

### MÓDULO 4: `src/demo.py`

#### 19. `carregar_detector_yolo(caminho_pesos: Path, device: str) -> YOLO`
- **O que faz:** Carrega os pesos e aloca a rede no hardware disponível (`cuda` se houver placa NVIDIA, ou `cpu`).
- **Por que existe:** Garante que a aplicação funcione em qualquer computador da faculdade sem quebrar.

#### 20. `determinar_tipo_fonte(fonte_str: str) -> tuple`
- **O que faz:** Checa se a entrada é um dígito (webcam), arquivo de vídeo (`.mp4`) ou foto (`.jpg`).
- **Por que existe:** Permite que um único script atenda a qualquer modalidade de demonstração.

#### 21. `executar_inferencia_frame(modelo, frame, conf, iou, imgsz) -> tuple`
- **O que faz:** Executa a predição na matriz de imagem e cronometra milissegundos puros de inferência.
- **Por que existe:** Isola a medição de tempo do modelo sem somar o tempo gasto para desenhar no OpenCV.

#### 22. `desenhar_deteccoes_frame(frame: np.ndarray, deteccoes: list) -> np.ndarray`
- **O que faz:** Desenha os retângulos ciano e etiquetas com classe e confiança formatada.
- **Por que existe:** Gera contraste visual nítido para projeção em telão ou monitor de sala.

#### 23. `desenhar_hud_informacoes(frame, total_buracos, fps, latencia_ms) -> np.ndarray`
- **O que faz:** Renderiza a barra escura superior com contador de buracos e taxa de quadros por segundo (FPS).
- **Por que existe:** Cumpre a exigência explícita do enunciado da matéria de exibir contagem e FPS ao vivo.

#### 24. `processar_modo_imagem(...)` e `processar_modo_stream(...)`
- **O que faz:** Controlam os loops de exibição de imagem estática ou fluxo contínuo de vídeo/webcam com tratamento de tecla `q` e `s`.
- **Por que existe:** Separa a lógica de fluxo para não travar o pipeline.

---

## 📌 PARTE 4: AS 20 PERGUNTAS MAIS PROVÁVEIS DA BANCA (COM RESPOSTAS)

### Perguntas sobre Problema e Alinhamento CitizenTech
**1. Por que este projeto é classificado como CitizenTech e não apenas uma solução governamental tradicional?**  
*Resposta:* "Porque a tecnologia é pensada a partir da vivência do cidadão comum e utiliza dispositivos acessíveis do dia a dia (smartphones e câmeras veiculares de baixo custo) para democratizar o monitoramento do espaço público. Em vez de depender de burocracias ou ouvidorias lentas, qualquer cidadão ou motorista de transporte público atua como agente colaborativo da zeladoria urbana."

**2. Como a sua solução se conecta especificamente aos ODS 11 e 9 da ONU?**  
*Resposta:* "No ODS 11 (Cidades Sustentáveis, Meta 11.2), a detecção precoce de buracos reduz diretamente o risco de acidentes graves com motociclistas, ciclistas e pedestres. No ODS 9 (Infraestrutura, Meta 9.1), permite à gestão pública realizar manutenção preventiva baseada em dados, otimizando o gasto tributário e tornando o pavimento mais resiliente."

**3. Qual é o diferencial prático de usar Visão Computacional em vez de acelerômetros em smartphones?**  
*Resposta:* "Acelerômetros só registram o impacto mecânico *depois* que a roda já caiu no buraco, gerando desconforto e risco de acidente imediato. A Visão Computacional atua de forma preditiva e ótica: ela enxerga a descontinuidade a metros de distância antes que o veículo passe por ela, além de fornecer o registro visual objetivo do dano."

### Perguntas sobre Dados e Dataset
**4. Qual dataset você utilizou, qual a licença e quantas imagens ele possui?**  
*Resposta:* "Utilizamos o Pothole Detection Dataset do Intel Unnati Training Program, hospedado no Roboflow Universe sob licença MIT (permissiva para pesquisa acadêmica). Ele contém mais de 2.700 imagens veiculares reais. Nosso conjunto de validação independente utilizou 491 imagens contendo 1.079 anotações."

**5. Por que foi importante o dataset ter classes separadas para 'Sewer Cover' e 'Drain Hole' além de 'Pothole'?**  
*Resposta:* "Para evitar que o modelo sofra de falsos alarmes crônicos. Em vias urbanas, tampas de bueiro metálicas e ralos pluviais têm desníveis e sombras circulares parecidas com buracos. Se rotulássemos tudo como uma classe só, a prefeitura receberia alertas de manutenção onde existe apenas um bueiro regular da companhia de saneamento."

**6. Como você garantiu que não houve vazamento de dados (data leakage)?**  
*Resposta:* "Mantivemos a separação estrita dos dados em splits isolados de treino e validação desde o download, e o script `src/dataset.py` validou a integridade e paridade dos arquivos sem compartilhamento de imagens entre os grupos."

### Perguntas sobre Arquitetura e Modelagem
**7. Por que você escolheu o YOLOv8n (nano) em vez de versões maiores como YOLOv8m ou YOLOv8x?**  
*Resposta:* "Por um compromisso de engenharia voltado a Edge Computing. O YOLOv8n possui apenas 3,13 milhões de parâmetros e 8,6 GFLOPs (~6,5 MB no disco). Uma versão maior consumiria muita memória e bateria, inviabilizando a execução em tempo real em smartphones ou CPUs sem placa de vídeo dedicada."

**8. O que é Transfer Learning e como ele foi aplicado no seu treinamento?**  
*Resposta:* "É a técnica de iniciar a rede neural a partir de pesos previamente treinados em uma grande base de dados (o Microsoft COCO com 80 classes universais), em vez de começar com pesos aleatórios. Assim, as primeiras camadas convolucionais já entram sabendo reconhecer bordas, curvas e gradientes de luz, necessitando apenas de fine-tuning para aprender a geometria de buracos no asfalto."

**9. O que significa o YOLO ser uma arquitetura 'Single-Shot'?**  
*Resposta:* "Significa que ele prevê as caixas delimitadoras e as probabilidades de classe em uma única passada (*forward pass*) da imagem pela rede neural convolucional. Diferente de redes de dois estágios (como Faster R-CNN), ele não gera regiões candidatas separadamente, o que o torna de 5 a 10 vezes mais rápido."

**10. Quais funções de perda (losses) compõem a função de custo do YOLOv8?**  
*Resposta:* "Uma perda combinada com três termos: CIoU Loss para penalizar erros de proporção e localização espacial da caixa; Binary Cross-Entropy (BCE) para a classificação das categorias; e Distribution Focal Loss (DFL) para refinar com precisão os contornos e limites irregulares das bordas."

### Perguntas sobre Métricas e Resultados
**11. Quais foram as métricas finais obtidas no seu modelo?**  
*Resposta:* "No conjunto de validação com 491 imagens e 1.079 instâncias, obtivemos 91,96% de Precisão, 85,37% de Recall, 89,02% de mAP@50 e 58,56% de mAP@50-95, com inferência média de 4,8 ms em GPU e cerca de 42 ms em CPU local."

**12. Explique a diferença entre Precisão e Recall com base nos resultados do seu projeto.**  
*Resposta:* "A Precisão de 91,96% mede a confiabilidade dos alarmes: quando o modelo diz que viu um buraco, ele está certo em quase 92% das vezes, evitando despachar equipes de tapa-buraco à toa. O Recall de 85,37% mede a cobertura: de cada 100 buracos existentes na via, o sistema consegue identificar 85 deles."

**13. Por que o Recall (85,37%) deu menor que a Precisão (91,96%)?**  
*Resposta:* "Porque ajustamos o limiar de confiança para 0.25 para priorizar predições seguras. Além disso, buracos muito rasos (menos de 2 a 3 cm) ou preenchidos com água barrenta pós-chuva perdem o degrau de sombra que indica profundidade, fazendo com que a rede opte por não disparar a detecção para evitar alarme falso."

**14. O que é o mAP@50 e por que ele é a métrica principal em detecção de objetos?**  
*Resposta:* "O mAP@50 é a Média da Precisão Média calculada sob um limiar de sobreposição IoU de 50%. Ele sintetiza em um único número a qualidade da curva Precisão $\times$ Recall em todos os níveis de confiança. Nosso resultado de 89,02% comprova alta estabilidade do detector."

### Perguntas sobre Análise Crítica e Diagnóstico de Erros
**15. Quais são os principais casos em que o seu modelo gera Falsos Positivos (alarmes falsos)?**  
*Resposta:* "Principalmente sombras recortadas de copas de árvores em dias de sol a pino, pois a sombra irregular sobre o asfalto cinza claro cria um gradiente de borda que simula a cavidade escura de um buraco. Outro caso são remendos recentes de asfalto novo com piche muito preto em contraste com pistas velhas."

**16. Quais são os principais casos em que o seu modelo gera Falsos Negativos (omissões)?**  
*Resposta:* "Buracos inundados por água de chuva ou lama, porque a lâmina d'água atua como espelho ou nivela a superfície com a pista, eliminando a percepção monocular de profundidade; e desgastes superficiais rasos que não possuem relevo geométrico expressivo."

**17. Como esses erros poderiam ser mitigados em uma versão 2.0 da solução?**  
*Resposta:* "Através de rastreamento temporal (*tracking*) entre quadros consecutivos (uma sombra se move diferente de um buraco fixo na via), fusão sensorial com acelerômetros do veículo e expansão do dataset com exemplos rotulados de chuva e pavimento molhado."

### Perguntas sobre Código, Arquitetura e Engenharia de Software
**18. Como o código está estruturado para garantir manutenção e clareza?**  
*Resposta:* "Seguimos o princípio da responsabilidade única (Single Responsibility Principle): cada função tem uma docstring detalhada em português explicando o que faz, parâmetros e o porquê de existir. O pipeline é desacoplado em quatro módulos: `dataset.py` (validação e YAML), `evaluate.py` (métricas oficiais e JSON), `analise_erros.py` (diagnóstico qualitativo de FPs/FNs) e `demo.py` (visualização em tempo real)."

**19. Como funciona o cálculo de FPS e contagem de buracos em tempo real no `demo.py`?**  
*Resposta:* "A inferência roda quadro a quadro e cronometramos o tempo com `time.perf_counter()`. Para o FPS não oscilar abruptamente na tela, aplicamos um filtro de suavização exponencial ($FPS = 0.9 \cdot FPS_{\text{ant}} + 0.1 \cdot FPS_{\text{inst}}$). A contagem é a soma dos bounding boxes com confiança acima de 0.25 desenhados no cabeçalho translúcido (HUD)."

**20. Como seria a implantação desta solução em larga escala em uma cidade como São Paulo?**  
*Resposta:* "Adotando arquitetura de Edge AI: a inferência roda localmente no dispositivo embarcado do veículo. Em vez de enviar vídeos pesados que sobrecarregariam redes móveis, o dispositivo transmite apenas pequenas mensagens JSON (<1 KB) com coordenadas GPS, timestamp e severidade para uma API em nuvem, que agrupa detecções repetidas em um raio de 5 metros e plota mapas de calor para a rota das equipes de manutenção."

---

## 📌 PARTE 5: CHECKLIST FINAL ANTES DE ENTRAR NA SALA

- [ ] Repositório clonado e funcionando localmente.
- [ ] `docs/relatorio.pdf` salvo no computador e acessível no GitHub.
- [ ] Teste de comando executado previamente: `python src/demo.py --source examples/exemplo_buracos.jpg --weights models/best.pt`.
- [ ] Leitura prévia deste guia de estudo para fixar os números principais: **Precisão: 91,96%**, **Recall: 85,37%**, **mAP@50: 89,02%** e **FPS: ~24 a 111**.
