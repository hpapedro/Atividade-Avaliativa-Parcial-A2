"""
================================================================================
Arquivo Principal: main.py
Projeto: Sistema de Visão Computacional para Monitoramento de Ocupação de Salas
Tema: Computação Visual com OpenCV e YOLO (Opção 2)
Descrição:
    Ponto de entrada da aplicação. Permite executar o monitoramento de ocupação
    em tempo real a partir de uma webcam, arquivo de vídeo (.mp4, .avi) ou
    imagem estática (.jpg, .png), exibindo caixas delimitadoras, nível de
    confiança e estatísticas na tela.

Exemplos de uso no terminal:
    python main.py                           # Abre webcam padrão (índice 0) ou modo demo
    python main.py --source 0 --capacidade 5 # Monitora webcam com limite de 5 pessoas
    python main.py --source midia/sala.mp4   # Analisa um arquivo de vídeo
    python main.py --source midia/teste.jpg  # Analisa uma imagem e salva o resultado
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from detector import MonitorOcupacaoSala


def criar_parser() -> argparse.ArgumentParser:
    """
    Configura e retorna o parser de argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de Visão Computacional - Monitoramento de Ocupação de Salas com OpenCV e YOLOv8"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fonte de entrada: '0' para webcam, ou caminho para arquivo de vídeo/imagem (padrão: 0).",
    )
    parser.add_argument(
        "--capacidade",
        type=int,
        default=5,
        help="Capacidade máxima de pessoas permitida na sala (padrão: 5).",
    )
    parser.add_argument(
        "--confianca",
        type=float,
        default=0.45,
        help="Nível mínimo de confiança para detecção do YOLO entre 0.1 e 0.95 (padrão: 0.45).",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="yolov8n.pt",
        help="Modelo YOLO pré-treinado a ser utilizado (padrão: yolov8n.pt).",
    )
    parser.add_argument(
        "--preprocessar",
        action="store_true",
        help="Ativa o pré-processamento OpenCV (GaussianBlur + CLAHE) desde o início.",
    )
    parser.add_argument(
        "--salvar-saida",
        action="store_true",
        help="Salva automaticamente a imagem ou vídeo processado na pasta 'resultados/'.",
    )
    parser.add_argument(
        "--sem-janela",
        action="store_true",
        help="Executa o processamento sem abrir janela gráfica cv2.imshow (útil para testes automatizados).",
    )
    return parser


def processar_imagem_estatica(
    monitor: MonitorOcupacaoSala,
    caminho_imagem: Path,
    pasta_resultados: Path,
    sem_janela: bool = False,
) -> None:
    """
    Processa uma única imagem estática, exibe as detecções e salva o resultado.
    """
    imagem = cv2.imread(str(caminho_imagem))
    if imagem is None:
        print(f"[ERRO] Não foi possível carregar a imagem: {caminho_imagem}")
        sys.exit(1)

    frame_anotado, deteccoes, estatisticas = monitor.processar_frame(imagem)

    # Salva sempre o resultado de imagem estática na pasta resultados/
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    caminho_saida = pasta_resultados / f"resultado_{caminho_imagem.name}"
    cv2.imwrite(str(caminho_saida), frame_anotado)

    print("\n" + "=" * 65)
    print("RESULTADO DA ANÁLISE DE IMAGEM")
    print("=" * 65)
    print(f"Arquivo analisado   : {caminho_imagem}")
    print(f"Pessoas detectadas  : {estatisticas['quantidade_atual']} / {estatisticas['capacidade_maxima']}")
    print(f"Taxa de ocupação    : {estatisticas['taxa_ocupacao']:.1f}%")
    print(f"Status do ambiente  : {estatisticas['status_texto']}")
    print(f"Confiança média     : {estatisticas['confianca_media']:.1f}%")
    print(f"Área total ocupada  : {estatisticas['area_total_percentual']:.1f}% do quadro")
    print(f"Imagem salva em     : {caminho_saida}")
    print("=" * 65)

    if not sem_janela:
        titulo_janela = "Monitoramento de Ocupacao de Salas - OpenCV + YOLOv8"
        cv2.imshow(titulo_janela, frame_anotado)
        print("[INFO] Pressione qualquer tecla na janela da imagem para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def processar_video_ou_webcam(
    monitor: MonitorOcupacaoSala,
    fonte: str,
    pasta_resultados: Path,
    salvar_saida: bool = False,
    sem_janela: bool = False,
) -> None:
    """
    Captura frames continuamente de uma webcam ou arquivo de vídeo, executando
    a detecção e exibindo os resultados em tempo real.
    """
    # Se a fonte for um número inteiro (ex: '0'), abre a webcam correspondente
    indice_ou_caminho = int(fonte) if fonte.isdigit() else fonte
    captura = cv2.VideoCapture(indice_ou_caminho)

    if not captura.isOpened():
        print(f"\n[ERRO] Não foi possível abrir a fonte de vídeo/webcam '{fonte}'.")
        print("[DICA] Verifique se a webcam está conectada ou informe o caminho de um vídeo/imagem com --source.")
        sys.exit(1)

    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_origem = captura.get(cv2.CAP_PROP_FPS)
    if not fps_origem or fps_origem <= 0 or fps_origem > 120:
        fps_origem = 25.0

    gravador = None
    caminho_video_saida = None
    if salvar_saida:
        pasta_resultados.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        caminho_video_saida = pasta_resultados / f"monitoramento_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        gravador = cv2.VideoWriter(str(caminho_video_saida), fourcc, fps_origem, (largura, altura))

    print("\n" + "=" * 65)
    print("MONITORAMENTO EM TEMPO REAL INICIADO")
    print("=" * 65)
    print("Atalhos de teclado na janela do vídeo:")
    print("  [q] ou [ESC] : Encerrar o sistema")
    print("  [s]          : Salvar captura de tela (screenshot) na pasta 'resultados/'")
    print("  [p]          : Ativar/Desativar pré-processamento OpenCV (CLAHE)")
    print("  [+] / [-]    : Aumentar / Diminuir a capacidade máxima da sala")
    print("=" * 65 + "\n")

    titulo_janela = "Monitoramento de Ocupacao de Salas - OpenCV + YOLOv8"
    ultimo_resumo = {}

    try:
        while True:
            sucesso, frame = captura.read()
            if not sucesso or frame is None:
                print("[INFO] Fim da transmissão de vídeo.")
                break

            # Processa o frame atual com OpenCV + YOLO
            frame_anotado, _, estatisticas = monitor.processar_frame(frame)
            ultimo_resumo = estatisticas

            if gravador is not None:
                gravador.write(frame_anotado)

            if not sem_janela:
                cv2.imshow(titulo_janela, frame_anotado)
                tecla = cv2.waitKey(1) & 0xFF

                if tecla in (ord("q"), 27):  # 'q' ou ESC
                    print("[INFO] Encerrado pelo usuário.")
                    break
                elif tecla == ord("s"):  # Salvar screenshot para o relatório
                    pasta_resultados.mkdir(parents=True, exist_ok=True)
                    nome_print = f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    caminho_print = pasta_resultados / nome_print
                    cv2.imwrite(str(caminho_print), frame_anotado)
                    print(f"[INFO] Captura de tela salva em: {caminho_print}")
                elif tecla == ord("p"):  # Alternar pré-processamento OpenCV
                    monitor.aplicar_preprocessamento = not monitor.aplicar_preprocessamento
                    estado = "ATIVADO" if monitor.aplicar_preprocessamento else "DESATIVADO"
                    print(f"[INFO] Pré-processamento OpenCV (CLAHE + GaussianBlur): {estado}")
                elif tecla in (ord("+"), ord("=")):
                    monitor.capacidade_maxima += 1
                    print(f"[INFO] Capacidade máxima ajustada para: {monitor.capacidade_maxima} pessoas")
                elif tecla == ord("-"):
                    monitor.capacidade_maxima = max(1, monitor.capacidade_maxima - 1)
                    print(f"[INFO] Capacidade máxima ajustada para: {monitor.capacidade_maxima} pessoas")
    finally:
        captura.release()
        if gravador is not None:
            gravador.release()
        if not sem_janela:
            cv2.destroyAllWindows()

    # Exibe o relatório estatístico final da sessão no console
    if ultimo_resumo:
        print("\n" + "=" * 65)
        print("RESUMO ESTATÍSTICO DA SESSÃO")
        print("=" * 65)
        print(f"Total de quadros (frames) analisados : {monitor.total_frames_processados}")
        print(f"Capacidade máxima configurada        : {monitor.capacidade_maxima} pessoas")
        print(f"Pico máximo de ocupação registrado   : {ultimo_resumo['pico_ocupacao']} pessoas")
        print(f"Média de pessoas na sala             : {ultimo_resumo['media_ocupacao']:.2f} pessoas")
        print(f"Desempenho médio (FPS)               : {ultimo_resumo['fps']:.1f} quadros/s")
        if caminho_video_saida:
            print(f"Vídeo gravado salvo em               : {caminho_video_saida}")
        print("=" * 65)


def main() -> None:
    """
    Função principal que interpreta os argumentos e direciona o fluxo de execução.
    """
    parser = criar_parser()
    args = parser.parse_args()

    monitor = MonitorOcupacaoSala(
        modelo_path=args.modelo,
        capacidade_maxima=args.capacidade,
        confianca_minima=args.confianca,
        aplicar_preprocessamento=args.preprocessar,
    )

    pasta_resultados = Path("resultados")
    caminho_fonte = Path(args.source)

    # Verifica se a fonte informada é um arquivo de imagem estática
    extensoes_imagem = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if caminho_fonte.is_file() and caminho_fonte.suffix.lower() in extensoes_imagem:
        processar_imagem_estatica(
            monitor=monitor,
            caminho_imagem=caminho_fonte,
            pasta_resultados=pasta_resultados,
            sem_janela=args.sem_janela,
        )
    else:
        processar_video_ou_webcam(
            monitor=monitor,
            fonte=args.source,
            pasta_resultados=pasta_resultados,
            salvar_saida=args.salvar_saida,
            sem_janela=args.sem_janela,
        )


if __name__ == "__main__":
    main()

