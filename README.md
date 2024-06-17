Aqui está uma versão aprimorada do seu README:

# Reconhecimento da Linguagem LIBRAS e Outros Gestos da Mão

## Descrição do Projeto

Este projeto utiliza o MediaPipe para detectar e classificar gestos das mãos em tempo real utilizando uma câmera. Foi desenvolvido para capturar gestos e gravá-los em arquivos CSV para análise posterior. O sistema emprega um modelo pré-treinado para classificar os gestos das mãos e possui funcionalidades de detecção de mãos, processamento de gestos e exibição de resultados. Assim, criamos uma base de dados que reconhece a linguagem LIBRAS e outros 5 gestos adicionais.

### Gestos Reconhecidos
- Alfabeto completo em LIBRAS
- Outros gestos:
  - Relógio (Sentido Horário)
  - Tchau (Mexer a mão aberta 2 vezes para cada lado, começando pela direita)
  - Não (Mexer a mão e o dedo 2 vezes para cada lado, começando pela direita)
  - Oi (Sentido Horário)
  - Tudo Bem?

## Funcionalidades

- Detecção e rastreamento de mãos em tempo real.
- Classificação de gestos utilizando um modelo pré-treinado.
- Registro de gestos em arquivos CSV.
- Exibição das coordenadas dos pontos de referência das mãos e das conexões entre eles.
- Alternância entre diferentes modos de operação utilizando teclas específicas.
- Geração de dados a partir de imagens.
- Geração de dados pela prórpia webcam.
- Criação de modelos de gestos treinados.

## Requisitos

- Python 3.7+
- OpenCV
- Mediapipe
- NumPy
- Pandas

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```
2. Navegue até o diretório do projeto:
   ```bash
   cd seu-repositorio
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Estrutura do Projeto

- `app.py`: Script principal que executa o sistema de reconhecimento de gestos.
- `model/`: Diretório contendo o modelo e os arquivos relacionados à classificação de gestos.
- `draw.py`: Script para desenhar os pontos de referência e conexões nas imagens.

## Como Executar

1. Conecte uma câmera ao seu computador.
2. Execute o script principal:
   ```bash
   python app.py
   ```
3. Utilize as teclas especificadas para alternar entre os modos e registrar gestos.
4. A imagem capturada será exibida em uma janela com os pontos de referência e conexões desenhados, bem como informações sobre o gesto classificado e o modo atual.

### Seleção de Modos

A função `selecionar_modo(tecla, modo)` permite alternar entre diferentes modos de operação utilizando as seguintes teclas:

- **TAB (Tecla 9)**: Alterna para o modo 0 (Modo padrão).
- **VÍRGULA (Tecla 32)**: Alterna para o modo 1 (Registro de histórico de gestos).
- **ESPAÇO (Tecla 44)**: Alterna para o modo 2 (Histórico de gestos estáticos).

Além disso, as teclas de '0' a '9' e de 'a' a 'z' podem ser usadas para registrar gestos específicos:

- **'0' a '9'**: As teclas numéricas de '0' a '9' são convertidas para os números correspondentes.
- **'a' a 'z'**: As teclas alfabéticas de 'a' a 'z' são convertidas para os números de 10 a 35.

## Data_Base
Arquivo .csv gerado: [gesture_sign_history.csv](https://drive.google.com/file/d/1IsoZwXI1gz-sfuxFUAJP15IWeVUBybhv/view?usp=sharing) 
Imagens Externas: https://www.kaggle.com/datasets/williansoliveira/libras
  

## Funções Principais Do Main

### main()

A função principal que configura a câmera, carrega o modelo, lê os rótulos dos gestos e inicia o loop de captura de vídeo para detecção e classificação de gestos.

### selecionar_modo(tecla, modo)

Permite alternar entre diferentes modos de operação e registrar gestos específicos com base nas teclas pressionadas.

### calcular_lista_pontos(imagem, pontos_landmarks)

Converte coordenadas normalizadas em coordenadas de pixel para a imagem.

### pre_processar_pontos(historico_pontos)

Pré-processa os pontos históricos para normalização e preparação para a classificação.

### processar_historico_pontos(imagem, historico_pontos, modo)

Processa o histórico de pontos para normalização em relação ao ponto base.

### processamento_combinado(imagem, historico_pontos, modo)

Combina os resultados das funções de pré-processamento e processamento do histórico de pontos.

### gravar_csv(numero, modo, lista_gestos)

Grava os gestos registrados em um arquivo CSV.

### registrar_gestos(numero, modo, lista_gestos)

Registra os gestos em um arquivo Feather para análise posterior.

### draw_landmarks_on_image(image, landmark_list, connections, ...)

Desenha os pontos de referência e as conexões nas imagens capturadas.

### calcula_retangulo_delimitador(pontos_landmarks)

Calcula a caixa delimitadora para os pontos de referência das mãos.

### desenhar_retangulo_borda(usando_borda, imagem, borda, mao_dominante, texto_gesto_mao, modo, numero)

Desenha um retângulo ao redor da mão e exibe informações sobre a mão dominante e o gesto classificado.

## Funções Adicionais 

### Classificação
A classe `ClassificarGestos` utiliza o TensorFlow Lite para classificar gestos das mãos. Inicialmente, carrega um modelo TFLite especificado pelo caminho, define um limiar de confiança e um valor para resultados inválidos. No método `__call__`, a classe recebe pontos históricos da mão, define o tensor de entrada do modelo, processa a entrada com o interpretador, obtém o resultado da classificação e verifica se a confiança é suficiente. Se a confiança for inferior ao limiar, retorna um valor inválido; caso contrário, retorna o índice do gesto classificado.

### CreatData
O script principal utiliza MediaPipe para detectar e processar gestos das mãos em imagens. Inicialmente, configura parâmetros de confiança para detecção e rastreamento. Em seguida, inicializa o MediaPipe Hands para detectar uma mão por vez e define um histórico de coordenadas utilizando deque.

O código percorre um diretório de imagens, selecionando até 100 arquivos aleatórios de cada subpasta. Para cada imagem, realiza a detecção dos landmarks das mãos, atualiza o histórico de coordenadas e processa esses dados. Os resultados processados são então gravados em arquivos CSV para análise posterior. O mapeamento de letras para números é feito através de um dicionário específico.

A função main() coordena todo o processo, desde a configuração inicial até a gravação dos resultados, sendo executada quando o script é rodado.

### 


## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias e correções.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Referências

- MediaPipe Authors. (2020). MediaPipe Python (Version 2.0) [Source code]. GitHub. [Link](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/python)
- Stack Overflow. (2021). Create a rectangle around all the points returned from MediaPipe hand landmark detection. [Link](https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d)
- OpenCV Documentation. (2021). OpenCV-Python Tutorials. [Link](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- Google AI. MediaPipe Solutions Guide. [Link](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=pt-br)
- TensorFlow. (2021). TensorFlow Lite: Guia de inferência. [Link](https://www.tensorflow.org/lite/guide/inference?hl=pt-br#load_and_run_a_model_in_python)
- TensorFlow. (2021). TensorFlow Bibliography. [Link](https://www.tensorflow.org/about/bib?hl=pt-br)

---

Este README fornece uma visão geral do projeto, suas funcionalidades, instruções de instalação e execução, bem como uma explicação detalhada sobre os modos de operação e as funções principais do código.
