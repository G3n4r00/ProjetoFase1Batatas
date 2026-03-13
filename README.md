# Projeto de Inspeção de Batatas com Visão Computacional

Este projeto utiliza inteligência artificial e visão computacional para classificar batatas como "Saudáveis" ou "Doentes" com base em imagens. O sistema é construído usando TensorFlow/Keras com o modelo MobileNetV2 pré-treinado, incorporando técnicas de data augmentation e fine-tuning para melhorar a precisão.

## Funcionamento Geral

O projeto consiste em três componentes principais:

1. **Treinamento do Modelo**: Scripts para treinar um modelo de classificação de imagens usando transfer learning com MobileNetV2.
2. **Classificação em Lote**: Script para processar múltiplas imagens e gerar relatórios em CSV.
3. **Aplicação Web**: Interface Streamlit para upload e classificação individual de imagens com geração de relatórios CSV.

### Arquitetura do Modelo

- **Base**: MobileNetV2 pré-treinado no ImageNet
- **Pré-processamento**: Redimensionamento para 224x224, data augmentation (flip, rotation, zoom), normalização
- **Camadas Finais**: Global Average Pooling, Dropout, Dense (128 unidades), Dense (2 unidades com softmax)
- **Técnicas**: Transfer learning com fine-tuning opcional

## Pré-requisitos

- Python 3.12
- TensorFlow 2.x
- Streamlit
- OpenCV
- Pillow (PIL)
- NumPy
- Pandas (para manipulação de dados no app web)

## Instalação

1. Clone ou baixe este repositório.
2. Instale as dependências:

```bash
pip install tensorflow streamlit opencv-python pillow numpy pandas
```

3. Certifique-se de que os arquivos de modelo estão presentes (modelo_batatas_finetuned.keras ou similar).

## Preparação do Dataset

Como as imagens não são incluídas no repositório, você deve preparar seu próprio dataset:

1. **Crie a estrutura de pastas**:
   ```
   dataset/
   ├── Doente/     # Coloque aqui imagens de batatas doentes
   └── Saudavel/   # Coloque aqui imagens de batatas saudáveis
   ```

2. **Para testes**: Crie uma pasta `imagens_teste/` com imagens adicionais para classificação em lote.

Foi utilizado o dataset público: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

## Estrutura do Projeto

```
ProjetoFase1Batatas/
├── app.py                          # Aplicação web Streamlit
├── treinamento_com_finetuning.py   # Script de treinamento com fine-tuning
├── classificacao_exporte.py        # Classificação em lote e exportação CSV
├── modelo_batatas_finetuned.keras  # Modelo treinado com fine-tuning
├── dataset/                        # [CRIAR] Dados de treinamento (não incluído)
│   ├── Doente/                     # [CRIAR] Imagens de batatas doentes
│   └── Saudavel/                   # [CRIAR] Imagens de batatas saudáveis
├── imagens_teste/                  # [CRIAR] Imagens para teste/classificação
└── .gitignore                      # Arquivos a ignorar no Git
```

## Como Usar

### 1. Treinamento do Modelo

#### Treinamento com Fine-Tuning

Execute o script `treinamento_com_finetuning.py`:

```bash
python treinamento_com_finetuning.py
```

Este script:
- **Fase 1**: Treina apenas as camadas finais por 5 épocas
- **Fase 2**: Descongela as últimas camadas do MobileNetV2 e faz fine-tuning por até 15 épocas
- Usa callbacks para early stopping e redução de learning rate
- Salva o modelo como `modelo_batatas_finetuned.keras`

### 2. Classificação em Lote

Para classificar múltiplas imagens e gerar um relatório:

```bash
python classificacao_exporte.py
```

Este script:
- Processa todas as imagens na pasta `imagens_teste/`
- Classifica cada imagem usando o modelo `modelo_batatas_finetuned.keras`
- **Abre uma janela para o usuário escolher onde salvar o relatório CSV**
- Gera um relatório CSV com ID da imagem, status e grau de confiança

### 3. Aplicação Web

Para usar a interface web interativa:

```bash
streamlit run app.py
```

A aplicação permite:
- Upload de uma imagem (JPG, JPEG, PNG)
- Exibição da imagem carregada
- Processamento e diagnóstico (Saudável/Doente)
- Exibição do grau de confiança com barra de progresso
- **Acumulação de múltiplas inspeções em um relatório**
- **Geração e download de relatório CSV** com todos os resultados

## Dataset

O dataset deve estar organizado na pasta `dataset/` com duas subpastas:
- `Doente/`: Imagens de batatas doentes
- `Saudavel/`: Imagens de batatas saudáveis

## Resultados e Métricas

Baseado nos relatórios gerados, o modelo alcança altos níveis de confiança:
- Batatas saudáveis: Geralmente >90% de confiança
- Batatas doentes: Geralmente >95% de confiança
- Alguns casos de baixa confiança indicam necessidade de mais dados de treinamento

## Personalização

### Modificar Classes
Para adicionar mais classes (ex: "Podre", "Machucada"), edite:
- As pastas no `dataset/`
- O número de unidades na última camada Dense (de 2 para N classes)
- A lista `class_names` nos scripts

### Ajustar Hiperparâmetros
- **Épocas**: Modifique em `model.fit()`
- **Learning Rate**: Ajuste no `Adam()`
- **Data Augmentation**: Adicione/remova camadas em `data_augmentation`

### Usar Outro Modelo Base
Substitua `MobileNetV2` por outros modelos como `ResNet50`, `VGG16`, etc.

## Time

Para dúvidas ou sugestões, entre em contato com o desenvolvedor do projeto.
