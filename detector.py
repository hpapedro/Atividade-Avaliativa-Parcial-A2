"""
================================================================================
Módulo: detector.py
Projeto: Sistema de Visão Computacional para Monitoramento de Ocupação de Salas
Tema: Computação Visual com OpenCV e YOLO (Opção 2)
Descrição:
    Implementa a classe responsável pelo pré-processamento de imagem com OpenCV,
    inferência com o modelo pré-treinado YOLOv8, desenho das caixas delimitadoras
    (bounding boxes), exibição da classe e nível de confiança, e cálculo de
    estatísticas de ocupação em tempo real.
================================================================================
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO


class MonitorOcupacaoSala:
    """
    Classe principal que encapsula a detecção de pessoas via YOLO e o
    processamento visual via OpenCV para monitoramento de ocupação de ambientes.
    """

    # No dataset COCO (utilizado pelo modelo YOLO pré-treinado), a classe 0 corresponde a 'person' (pessoa)
    CLASSE_PESSOA_COCO = 0

    def __init__(
        self,
        modelo_path: str = "yolov8n.pt",
        capacidade_maxima: int = 5,
        confianca_minima: float = 0.45,
        aplicar_preprocessamento: bool = False,
    ) -> None:
        """
        Inicializa o monitor de ocupação de salas.

        Parâmetros:
            modelo_path (str): Nome ou caminho do modelo YOLO pré-treinado (padrão: yolov8n.pt).
            capacidade_maxima (int): Limite máximo de pessoas permitido na sala/ambiente.
            confianca_minima (float): Limiar mínimo de confiança (0.0 a 1.0) para validar uma detecção.
            aplicar_preprocessamento (bool): Define se o filtro de melhoria de contraste (CLAHE) inicia ativo.
        """
        print(f"[INFO] Carregando modelo YOLO pré-treinado: {modelo_path}...")
        self.modelo = YOLO(modelo_path)
        self.capacidade_maxima = max(1, capacidade_maxima)
        self.confianca_minima = confianca_minima
        self.aplicar_preprocessamento = aplicar_preprocessamento

        # Histórico de métricas para estatísticas da sessão
        self.total_frames_processados: int = 0
        self.pico_ocupacao: int = 0
        self.historico_contagem: List[int] = []
        self.tempo_anterior: float = time.time()
        self.fps_atual: float = 0.0

        # Cria o objeto CLAHE do OpenCV para equalização adaptativa de histograma (melhoria de iluminação)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocessar_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Realiza o pré-processamento da imagem utilizando OpenCV antes da detecção.
        Aplica suavização leve (GaussianBlur) para redução de ruído de câmera e
        equalização adaptativa de contraste (CLAHE) no espaço de cor LAB.

        Parâmetros:
            frame (np.ndarray): Imagem original capturada pelo OpenCV (formato BGR).

        Retorno:
            np.ndarray: Imagem pré-processada no formato BGR.
        """
        if not self.aplicar_preprocessamento:
            return frame

        # 1. Filtro Gaussiano leve para reduzir ruído de alta frequência da webcam/vídeo
        frame_suavizado = cv2.GaussianBlur(frame, (3, 3), 0)

        # 2. Conversão de BGR para LAB para melhorar o contraste apenas no canal de luminância (L)
        lab = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_equalizado = self.clahe.apply(l_channel)
        lab_equalizado = cv2.merge((l_equalizado, a_channel, b_channel))

        # 3. Converte de volta para o espaço BGR padrão do OpenCV
        frame_melhorado = cv2.cvtColor(lab_equalizado, cv2.COLOR_LAB2BGR)
        return frame_melhorado

    def detectar_pessoas(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executa a inferência do modelo YOLO no frame atual, filtrando exclusivamente
        a classe 'person' (ID 0 no COCO) acima do limiar de confiança configurado.

        Parâmetros:
            frame (np.ndarray): Frame de vídeo ou imagem em formato BGR.

        Retorno:
            List[Dict[str, Any]]: Lista de dicionários contendo coordenadas da caixa
                                  delimitadora (bbox), classe, confiança e área em pixels.
        """
        # Executa a predição do YOLO filtrando apenas a classe 0 (pessoa)
        resultados = self.modelo.predict(
            source=frame,
            conf=self.confianca_minima,
            classes=[self.CLASSE_PESSOA_COCO],
            verbose=False,
        )

        deteccoes: List[Dict[str, Any]] = []
        altura_frame, largura_frame = frame.shape[:2]
        area_total_frame = float(altura_frame * largura_frame)

        # Extrai as caixas delimitadoras (bounding boxes) retornadas pelo YOLO
        for resultado in resultados:
            if resultado.boxes is None:
                continue

            for box in resultado.boxes:
                # Coordenadas x1, y1 (canto superior esquerdo) e x2, y2 (canto inferior direito)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confianca = float(box.conf[0].item())
                classe_id = int(box.cls[0].item())

                largura_box = max(0, x2 - x1)
                altura_box = max(0, y2 - y1)
                area_pixels = largura_box * altura_box
                percentual_area = (area_pixels / area_total_frame) * 100.0 if area_total_frame > 0 else 0.0

                deteccoes.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "classe_id": classe_id,
                        "classe_nome": "Pessoa",
                        "confianca": confianca,
                        "area_pixels": area_pixels,
                        "percentual_area": percentual_area,
                    }
                )

        return deteccoes

    def calcular_estatisticas(self, deteccoes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula métricas e estatísticas em tempo real a partir das detecções do frame.

        Parâmetros:
            deteccoes (List[Dict[str, Any]]): Lista de pessoas detectadas no frame atual.

        Retorno:
            Dict[str, Any]: Dicionário com contagem atual, taxa de ocupação, status do
                            ambiente, confiança média, área ocupada, pico e média histórica.
        """
        quantidade_atual = len(deteccoes)
        self.total_frames_processados += 1
        self.historico_contagem.append(quantidade_atual)

        # Mantém janela móvel de até 500 frames para média estável sem consumo excessivo de memória
        if len(self.historico_contagem) > 500:
            self.historico_contagem.pop(0)

        if quantidade_atual > self.pico_ocupacao:
            self.pico_ocupacao = quantidade_atual

        # Calcula FPS em tempo real
        agora = time.time()
        delta = agora - self.tempo_anterior
        if delta > 0:
            fps_instantaneo = 1.0 / delta
            self.fps_atual = (0.85 * self.fps_atual + 0.15 * fps_instantaneo) if self.fps_atual > 0 else fps_instantaneo
        self.tempo_anterior = agora

        # Calcula confiança média e soma da área ocupada pelas pessoas no frame
        if quantidade_atual > 0:
            confianca_media = float(np.mean([d["confianca"] for d in deteccoes])) * 100.0
            area_total_percentual = float(np.sum([d["percentual_area"] for d in deteccoes]))
        else:
            confianca_media = 0.0
            area_total_percentual = 0.0

        taxa_ocupacao = (quantidade_atual / self.capacidade_maxima) * 100.0
        media_historica = float(np.mean(self.historico_contagem))

        # Define o status e a cor do alerta (formato BGR do OpenCV) conforme a ocupação da sala
        if quantidade_atual > self.capacidade_maxima:
            status_texto = "ACIMA DO LIMITE (LOTADO)"
            cor_status = (0, 0, 235)  # Vermelho em BGR
        elif taxa_ocupacao >= 80.0:
            status_texto = "ATENCAO (QUASE LOTADO)"
            cor_status = (0, 180, 255)  # Amarelo/Laranja em BGR
        else:
            status_texto = "OCUPACAO NORMAL"
            cor_status = (40, 200, 40)  # Verde em BGR

        return {
            "quantidade_atual": quantidade_atual,
            "capacidade_maxima": self.capacidade_maxima,
            "taxa_ocupacao": taxa_ocupacao,
            "status_texto": status_texto,
            "cor_status": cor_status,
            "confianca_media": confianca_media,
            "area_total_percentual": area_total_percentual,
            "pico_ocupacao": self.pico_ocupacao,
            "media_ocupacao": media_historica,
            "fps": self.fps_atual,
        }

    def desenhar_resultados(
        self,
        frame: np.ndarray,
        deteccoes: List[Dict[str, Any]],
        estatisticas: Dict[str, Any],
    ) -> np.ndarray:
        """
        Desenha as caixas delimitadoras (bounding boxes), rótulos de classe e confiança,
        e o painel de estatísticas (HUD) diretamente no frame utilizando OpenCV.

        Parâmetros:
            frame (np.ndarray): Frame de imagem onde os elementos serão desenhados.
            deteccoes (List[Dict[str, Any]]): Pessoas detectadas pelo YOLO.
            estatisticas (Dict[str, Any]): Estatísticas calculadas para o frame atual.

        Retorno:
            np.ndarray: Frame anotado pronto para exibição em tempo real.
        """
        frame_anotado = frame.copy()
        altura, largura = frame_anotado.shape[:2]

        # 1. Desenha cada caixa delimitadora (bounding box) e seu rótulo de confiança
        for idx, det in enumerate(deteccoes, start=1):
            x1, y1, x2, y2 = det["bbox"]
            conf_pct = det["confianca"] * 100.0
            rotulo = f"{det['classe_nome']} #{idx}: {conf_pct:.1f}%"

            # Cor da bounding box: verde para alta confiança, ciano/amarelo para confiança moderada
            cor_box = (50, 220, 50) if conf_pct >= 70.0 else (0, 215, 255)

            # Retângulo principal da pessoa detectada
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), cor_box, 2)

            # Fundo preenchido para o texto da classe + nível de confiança
            (larg_texto, alt_texto), baseline = cv2.getTextSize(
                rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            y_topo_rotulo = max(y1, alt_texto + 10)
            cv2.rectangle(
                frame_anotado,
                (x1, y_topo_rotulo - alt_texto - 8),
                (x1 + larg_texto + 10, y_topo_rotulo + baseline - 2),
                cor_box,
                cv2.FILLED,
            )

            # Texto com classe identificada e nível de confiança (%)
            cv2.putText(
                frame_anotado,
                rotulo,
                (x1 + 5, y_topo_rotulo - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 2. Desenha o Painel Superior (HUD) semitransparente com as estatísticas da sala
        overlay = frame_anotado.copy()
        altura_painel = 135
        largura_painel = min(largura - 20, 440)
        cv2.rectangle(overlay, (10, 10), (10 + largura_painel, 10 + altura_painel), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.78, frame_anotado, 0.22, 0, frame_anotado)

        # Borda lateral colorida conforme o status da ocupação
        cor_status = estatisticas["cor_status"]
        cv2.rectangle(frame_anotado, (10, 10), (18, 10 + altura_painel), cor_status, cv2.FILLED)

        # Textos das métricas em tempo real no painel
        linhas_hud = [
            (f"MONITORAMENTO DE OCUPACAO (YOLOv8 + OpenCV)", (255, 255, 255), 0.52, 32),
            (
                f"Pessoas na Sala: {estatisticas['quantidade_atual']} / {estatisticas['capacidade_maxima']} "
                f"({estatisticas['taxa_ocupacao']:.0f}%)",
                (255, 255, 255),
                0.62,
                58,
            ),
            (f"Status: {estatisticas['status_texto']}", cor_status, 0.58, 84),
            (
                f"Confianca Media: {estatisticas['confianca_media']:.1f}% | "
                f"Pico: {estatisticas['pico_ocupacao']} | Media: {estatisticas['media_ocupacao']:.1f}",
                (210, 210, 210),
                0.46,
                108,
            ),
            (
                f"Area Detectada: {estatisticas['area_total_percentual']:.1f}% | "
                f"CLAHE: {'ON' if self.aplicar_preprocessamento else 'OFF'} | "
                f"FPS: {estatisticas['fps']:.1f}",
                (180, 230, 255),
                0.45,
                130,
            ),
        ]

        for texto, cor, escala, pos_y in linhas_hud:
            cv2.putText(
                frame_anotado,
                texto,
                (28, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                escala,
                cor,
                1 if escala < 0.55 else 2,
                cv2.LINE_AA,
            )

        # 3. Barra de progresso visual da ocupação no rodapé da imagem
        margem_barra = 12
        y_barra = altura - 18
        largura_total_barra = largura - (2 * margem_barra)
        cv2.rectangle(
            frame_anotado,
            (margem_barra, y_barra),
            (margem_barra + largura_total_barra, y_barra + 10),
            (50, 50, 50),
            cv2.FILLED,
        )

        proporcao = min(1.0, estatisticas["taxa_ocupacao"] / 100.0)
        largura_preenchida = int(largura_total_barra * proporcao)
        if largura_preenchida > 0:
            cv2.rectangle(
                frame_anotado,
                (margem_barra, y_barra),
                (margem_barra + largura_preenchida, y_barra + 10),
                cor_status,
                cv2.FILLED,
            )

        return frame_anotado

    def processar_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executa o pipeline completo em um único frame:
        1. Pré-processamento OpenCV
        2. Detecção de pessoas com YOLOv8
        3. Cálculo de estatísticas de ocupação
        4. Renderização das caixas delimitadoras e métricas na imagem

        Parâmetros:
            frame (np.ndarray): Frame original capturado.

        Retorno:
            Tuple contendo (frame_anotado, lista_de_deteccoes, dicionario_estatisticas).
        """
        frame_proc = self.preprocessar_frame(frame)
        deteccoes = self.detectar_pessoas(frame_proc)
        estatisticas = self.calcular_estatisticas(deteccoes)
        frame_anotado = self.desenhar_resultados(frame_proc, deteccoes, estatisticas)
        return frame_anotado, deteccoes, estatisticas

