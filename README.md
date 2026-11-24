# Sistema de Monitoramento de Ocupação de Salas e Detecção de Pessoas

Aplicação de **Visão Computacional em Tempo Real** desenvolvida em Python utilizando **OpenCV** e **YOLOv8 (You Only Look Once)** para identificar, contar e monitorar a ocupação de pessoas em salas e ambientes internos por meio de **webcam**, **vídeos** ou **imagens**.

---

## 1. Funcionalidades Implementadas (Opção 2 — OpenCV + YOLO)

- **Modelo YOLO Pré-treinado (`yolov8n.pt`):** Detecção em tempo real da classe `person` (Pessoa) utilizando pesos pré-treinados no dataset COCO.
- **Múltiplas Fontes de Entrada:** Suporte a **webcam ao vivo**, **arquivos de vídeo (`.mp4`, `.avi`)** e **imagens estáticas (`.jpg`, `.png`)**.
- **Pré-processamento com OpenCV:** Filtro de suavização Gaussiana (`cv2.GaussianBlur`) e Equalização Adaptativa de Histograma (`cv2.createCLAHE` no espaço de cor LAB) para melhorar a detecção em ambientes com iluminação irregular.
- **Caixas Delimitadoras (*Bounding Boxes*):** Desenho em tempo real ao redor de cada pessoa identificada, com código de cores dinâmico.
- **Classe e Nível de Confiança:** Exibição do rótulo `Pessoa #N: XX.X%` no topo de cada caixa delimitadora.
- **Painel de Estatísticas em Tempo Real (HUD):**
  - Quantidade atual de pessoas na sala vs. capacidade máxima (`Pessoas na Sala: X / Y`).
  - Taxa percentual de ocupação (`%`) e barra de progresso no rodapé.
  - Alerta visual de status: **OCUPAÇÃO NORMAL** (Verde), **ATENÇÃO - QUASE LOTADO** (Amarelo) e **ACIMA DO LIMITE - LOTADO** (Vermelho).
  - Confiança média das detecções (`%`), percentual de área ocupada no quadro, pico máximo de ocupação, média histórica da sessão e taxa de quadros por segundo (**FPS**).

---

## 2. Estrutura do Projeto

```text
├── main.py             # Ponto de entrada principal (CLI para webcam, vídeo ou imagem)
├── detector.py         # Classe MonitorOcupacaoSala (OpenCV + YOLOv8 + Métricas)
├── requirements.txt    # Lista de bibliotecas e dependências do projeto
└── README.md           # Instruções de instalação e execução
```

---

## 3. Pré-requisitos e Instalação

Certifique-se de ter o **Python 3.9 ou superior** instalado. No terminal (PowerShell ou CMD), dentro da pasta do projeto, execute:

```bash
pip install -r requirements.txt
```

> **Nota:** Na primeira execução, a biblioteca `ultralytics` fará o download automático do modelo pré-treinado `yolov8n.pt` (~6 MB).

---

## 4. Como Executar o Projeto

### A) Executar em Tempo Real com a Webcam
Para abrir a webcam padrão do computador (índice `0`) com capacidade máxima configurada para 5 pessoas:

```bash
python main.py --source 0 --capacidade 5
```

### B) Executar em um Arquivo de Vídeo
Para processar um arquivo de vídeo `.mp4`:

```bash
python main.py --source caminho/do/video.mp4 --capacidade 5
```

Para também **gravar o vídeo processado** na pasta `resultados/`:

```bash
python main.py --source caminho/do/video.mp4 --capacidade 5 --salvar-saida
```

### C) Executar em uma Imagem Estática
Para analisar uma imagem e salvar o resultado anotado na pasta `resultados/`:

```bash
python main.py --source caminho/da/imagem.jpg --capacidade 5
```

---

## 5. Atalhos Interativos Durante a Execução

Com a janela do OpenCV aberta durante a execução de vídeo ou webcam, utilize as seguintes teclas:

| Tecla | Ação |
| :---: | :--- |
| **`q`** ou **`ESC`** | Encerra o monitoramento e exibe o resumo estatístico no terminal |
| **`s`** | Salva uma captura de tela (*screenshot*) do quadro atual na pasta `resultados/` |
| **`p`** | Ativa / Desativa ao vivo o pré-processamento de contraste do OpenCV (`CLAHE`) |
| **`+`** | Aumenta em `+1` a capacidade máxima da sala em tempo real |
| **`-`** | Diminui em `-1` a capacidade máxima da sala em tempo real |
