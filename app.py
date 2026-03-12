import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Configurações da Página Web
st.set_page_config(page_title="Inspeção de Batatas Utilizando Visão Computacional", page_icon="🥔", layout="centered")

st.title("🥔 Classificador de Batatas por IA")
st.write("Faça o upload de uma foto para o sistema identificar se a batata está saudável ou doente.")
st.markdown("---")

# 2. Carregar o Modelo (O cache evita que o modelo seja recarregado a cada clique)
@st.cache_resource
def carregar_modelo():
    # Certifique-se de que o nome do arquivo bate com o modelo que você treinou
    return load_model('modelo_batatas_finetuned.keras')

try:
    model = carregar_modelo()
    class_names = ['Doente', 'Saudável']
except Exception as e:
    st.error(f"Erro ao carregar o modelo. Verifique se o arquivo .keras está na pasta. Erro: {e}")
    st.stop()

# 3. Componente de Upload de Imagem
uploaded_file = st.file_uploader("Escolha a foto da batata...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ler e exibir a imagem na tela
    image = Image.open(uploaded_file).convert('RGB')
    
    # Criar duas colunas para o layout ficar mais bonito
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Imagem em Análise', use_container_width=True)
        
    with col2:
        st.markdown("### Processando...")
        with st.spinner('A IA está analisando a textura...'):
            # 4. Pré-processamento exato que o modelo exige
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized)
            image_array = np.expand_dims(image_array, axis=0)
            
            # 5. Predição
            prediction = model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            classe_detectada = class_names[index]
            confianca = prediction[0][index] * 100
            
            # 6. Exibição do Resultado
            st.markdown("### Resultado Final")
            if classe_detectada == 'Saudável':
                st.success(f"**Diagnóstico:** {classe_detectada} 🟢")
            else:
                st.error(f"**Diagnóstico:** {classe_detectada} 🔴")
                
            st.info(f"**Grau de Confiança:** {confianca:.1f}%")
            st.progress(int(confianca))