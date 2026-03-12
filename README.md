# Projeto de Inspeção de Batatas com Visão Computacional

Este projeto utiliza inteligência artificial e visão computacional para classificar batatas como "Saudáveis" ou "Doentes" com base em imagens. O sistema é construído usando TensorFlow/Keras com o modelo MobileNetV2 pré-treinado, incorporando técnicas de data augmentation e fine-tuning para melhorar a precisão.

## Funcionamento Geral

O projeto consiste em três componentes principais:

1. **Treinamento do Modelo**: Scripts para treinar um modelo de classificação de imagens usando transfer learning com MobileNetV2.
2. **Classificação em Lote**: Script para processar múltiplas imagens e gerar relatórios em CSV.
3. **Aplicação Web**: Interface Streamlit para upload e classificação individual de imagens.

### Arquitetura do Modelo

- **Base**: MobileNetV2 pré-treinado no ImageNet
- **Pré-processamento**: Redimensionamento para 224x224, data augmentation (flip, rotation, zoom), normalização
- **Camadas Finais**: Global Average Pooling, Dropout, Dense (128 unidades), Dense (2 unidades com softmax)
- **Técnicas**: Transfer learning com fine-tuning opcional

## Pré-requisitos

- Python 3.8 ou superior
- TensorFlow 2.x
- Streamlit
- OpenCV
- Pillow (PIL)
- NumPy

## Instalação

1. Clone ou baixe este repositório.
2. Instale as dependências:

```bash
pip install tensorflow streamlit opencv-python pillow numpy
```

3. Certifique-se de que os arquivos de modelo estão presentes (modelo_batatas_finetuned.keras ou similar).

## Estrutura do Projeto

```
ProjetoFase1Batatas/
├── app.py                          # Aplicação web Streamlit
├── treinamento_modelo.py           # Script de treinamento básico
├── treinamento_com_finetuning.py   # Script de treinamento com fine-tuning
├── classificacao_exporte.py        # Classificação em lote e exportação CSV
├── modelo_batatas_finetuned.keras  # Modelo treinado com fine-tuning
├── modelo_batatas_v1.h5            # Modelo versão 1
├── modelo_batatas_v2.h5            # Modelo versão 2
├── modelo_batatas_v3.keras         # Modelo versão 3
├── relatorio_inspecao_batatas.csv  # Relatório de inspeção
├── relatorio_inspecao_batatas_v2.csv # Relatório versão 2
├── dataset/                        # Dados de treinamento
│   ├── Doente/                     # Imagens de batatas doentes
│   └── Saudavel/                   # Imagens de batatas saudáveis
└── imagens_teste/                  # Imagens para teste/classificação
```

## Como Usar

### 1. Treinamento do Modelo

#### Treinamento Básico (Sem Fine-Tuning)

Execute o script `treinamento_modelo.py`:

```bash
python treinamento_modelo.py
```

Este script:
- Carrega o dataset da pasta `dataset/`
- Aplica data augmentation
- Treina o modelo por 12 épocas
- Salva o modelo como `modelo_batatas_v3.keras`

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
- Gera um relatório CSV com ID da imagem, status e grau de confiança
- Exemplo de saída: `relatorio_inspecao_batatas_v2.csv`

### 3. Aplicação Web

Para usar a interface web interativa:

```bash
streamlit run app.py
```

A aplicação:
- Permite upload de uma imagem (JPG, JPEG, PNG)
- Exibe a imagem carregada
- Processa a imagem e mostra o diagnóstico (Saudável/Doente)
- Mostra o grau de confiança com uma barra de progresso

## Dataset

O dataset deve estar organizado na pasta `dataset/` com duas subpastas:
- `Doente/`: Imagens de batatas doentes
- `Saudavel/`: Imagens de batatas saudáveis

As imagens são divididas automaticamente em 80% treinamento e 20% validação.

## Modelos Disponíveis

- `modelo_batatas_finetuned.keras`: Modelo com fine-tuning (recomendado)
- `modelo_batatas_v3.keras`: Modelo básico sem fine-tuning
- `modelo_batatas_v1.h5` e `modelo_batatas_v2.h5`: Versões anteriores

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

## Troubleshooting

- **Erro ao carregar modelo**: Verifique se o arquivo .keras existe e está no formato correto
- **Baixa acurácia**: Treine por mais épocas ou adicione mais dados
- **Streamlit não inicia**: Instale com `pip install streamlit` e execute em ambiente virtual
- **Imagens não processadas**: Verifique formato (RGB) e tamanho (224x224)

## Contribuição

Para contribuir:
1. Faça fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Contato

Para dúvidas ou sugestões, entre em contato com o desenvolvedor do projeto.