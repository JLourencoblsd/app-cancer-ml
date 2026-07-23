import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="MedAI | Diagnóstico Preditivo",
    page_icon="🫁",
    layout="centered"
)

# =========================
# CSS CUSTOMIZADO (Visual Leve & Profissional)
# =========================
st.markdown("""
    <style>
    /* Estilo do Título Principal */
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .badge-status {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    /* Estilização dos Botões */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stFormSubmitButton > button:first-child {
        background-color: #1E88E5;
        color: white;
        width: 100%;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        padding: 10px;
    }
    div.stFormSubmitButton > button:first-child:hover {
        background-color: #1565C0;
        border-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# CARREGAMENTO DO MODELO (Memória Cache)
# =========================
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_cancer.pkl")

modelo = carregar_modelo()

# =========================
# ESTADO DE SESSÃO
# =========================
if "historico" not in st.session_state:
    st.session_state.historico = []

# =========================
# FUNÇÕES AUXILIARES
# =========================
def sim_nao(valor):
    return 2 if valor == "Sim" else 1

# =========================
# CABEÇALHO DO APP
# =========================
st.markdown('<p class="title-text">🫁 MedAI - Pulmonar</p>', unsafe_allow_html=True)
st.markdown('<span class="badge-status">🟢 Modelo de IA Ativo | Acurácia Clínica</span>', unsafe_allow_html=True)
st.write("Insira os parâmetros do paciente para gerar o relatório de probabilidade clínica.")

# =========================
# COLUNAS DO MODELO
# =========================
ordem_correta = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ",
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

# =========================
# FORMULÁRIO PRINCIPAL
# =========================
with st.form("form_paciente"):
    st.subheader("👤 Identificação")
    nome = st.text_input("Nome completo do paciente", placeholder="Ex: João da Silva")
    
    st.divider()
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Dados Gerais")
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        age = st.number_input("Idade", 1, 120, 30)

        st.markdown("### 🚬 Hábitos & Histórico")
        smoking = sim_nao(st.selectbox("Fuma?", ["Não", "Sim"]))
        alcohol = sim_nao(st.selectbox("Consome álcool?", ["Não", "Sim"]))
        yellow_fingers = sim_nao(st.selectbox("Dedos amarelos?", ["Não", "Sim"]))
        peer_pressure = sim_nao(st.selectbox("Pressão social?", ["Não", "Sim"]))
        anxiety = sim_nao(st.selectbox("Ansiedade?", ["Não", "Sim"]))

    with col2:
        st.markdown("### 🩺 Sintomas Clínicos")
        fatigue = sim_nao(st.selectbox("Fadiga frequente?", ["Não", "Sim"]))
        allergy = sim_nao(st.selectbox("Alergia?", ["Não", "Sim"]))
        wheezing = sim_nao(st.selectbox("Chiado no peito?", ["Não", "Sim"]))
        coughing = sim_nao(st.selectbox("Tosse persistente?", ["Não", "Sim"]))
        shortness = sim_nao(st.selectbox("Falta de ar?", ["Não", "Sim"]))
        swallowing = sim_nao(st.selectbox("Dificuldade para engolir?", ["Não", "Sim"]))
        chest_pain = sim_nao(st.selectbox("Dor no peito?", ["Não", "Sim"]))
        chronic_disease = sim_nao(st.selectbox("Possui doença crônica?", ["Não", "Sim"]))

    gender_val = 1 if gender == "Masculino" else 0

    st.write("")
    btn_analisar = st.form_submit_button("📊 Gerar Análise de Risco")

# =========================
# PROCESSAMENTO & RESULTADOS
# =========================
if btn_analisar:
    if nome.strip() == "":
        st.warning("⚠️ Por favor, informe o nome do paciente para continuar.")
    else:
        dados = pd.DataFrame([[ 
            gender_val, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol,
            coughing, shortness, swallowing, chest_pain
        ]], columns=ordem_correta)

        # Calculando predição
        prob = modelo.predict_proba(dados)[0][1]
        prob_percent = prob * 100
        resultado = 1 if prob >= 0.5 else 0
        status = "Elevado Risco Clínico" if resultado == 1 else "Baixo Risco Clínico"

        # Adicionando ao Histórico
        st.session_state.historico.append({
            "Nome": nome,
            "Risco Numérico": prob,
            "Risco (%)": f"{prob_percent:.1f}%",
            "Classificação": status
        })

        # Exibição do Resultado em Formato Dashboard
        st.divider()
        st.subheader("📊 Relatório de Diagnóstico")

        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Paciente", nome)
        with res_col2:
            st.metric("Probabilidade de Risco", f"{prob_percent:.1f}%")
        with res_col3:
            st.metric("Status Diagnóstico", "⚠️ Positivo" if resultado == 1 else "✅ Negativo")

        # Barra de Progresso
        st.progress(float(prob))

        if resultado == 1:
            st.error(f"**Atenção:** O modelo identificou um **{status}** para {nome}. Recomenda-se acompanhamento médico imediato.")
        else:
            st.success(f"**Resultado Estável:** O paciente {nome} apresenta **{status}** com base nos sintomas informados.")

# =========================
# SEÇÃO DE HISTÓRICO
# =========================
st.divider()
st.subheader("📋 Histórico de Pacientes Analisados")

if len(st.session_state.historico) > 0:
    df_hist = pd.DataFrame(st.session_state.historico)
    
    # Exibição estilizada da tabela com barra visual integrada
    st.dataframe(
        df_hist[["Nome", "Risco Numérico", "Classificação"]],
        column_config={
            "Risco Numérico": st.column_config.ProgressColumn(
                "Nível de Risco",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Limpar Histórico de Consultas"):
        st.session_state.historico = []
        st.rerun()
else:
    st.info("Nenhuma análise realizada nesta sessão.")
