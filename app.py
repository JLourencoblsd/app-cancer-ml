import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Predição de Câncer de Pulmão",
    page_icon="🫁",
    layout="centered"
)

# =========================
# MODELO
# =========================
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_cancer.pkl")

modelo = carregar_modelo()

# =========================
# ESTADO (LISTA DE PACIENTES)
# =========================
if "historico" not in st.session_state:
    st.session_state.historico = []

# =========================
# FUNÇÕES
# =========================
def sim_nao(valor):
    return 2 if valor == "Sim" else 1

# =========================
# TÍTULO
# =========================
st.title("🫁 Detector de Risco de Câncer")
st.write("Preencha os dados abaixo")

# =========================
# NOME
# =========================
nome = st.text_input("Nome do paciente")

st.divider()

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
    age = st.number_input("Idade", 1, 120, 30)

    smoking = sim_nao(st.selectbox("Fuma?", ["Não", "Sim"]))
    yellow_fingers = sim_nao(st.selectbox("Dedos amarelos?", ["Não", "Sim"]))
    anxiety = sim_nao(st.selectbox("Ansiedade?", ["Não", "Sim"]))
    peer_pressure = sim_nao(st.selectbox("Pressão social?", ["Não", "Sim"]))
    chronic_disease = sim_nao(st.selectbox("Doença crônica?", ["Não", "Sim"]))

with col2:
    fatigue = sim_nao(st.selectbox("Fadiga?", ["Não", "Sim"]))
    allergy = sim_nao(st.selectbox("Alergia?", ["Não", "Sim"]))
    wheezing = sim_nao(st.selectbox("Chiado?", ["Não", "Sim"]))
    alcohol = sim_nao(st.selectbox("Consome álcool?", ["Não", "Sim"]))
    coughing = sim_nao(st.selectbox("Tosse?", ["Não", "Sim"]))
    shortness = sim_nao(st.selectbox("Falta de ar?", ["Não", "Sim"]))
    swallowing = sim_nao(st.selectbox("Dificuldade para engolir?", ["Não", "Sim"]))
    chest_pain = sim_nao(st.selectbox("Dor no peito?", ["Não", "Sim"]))

gender = 1 if gender == "Masculino" else 0

# =========================
# COLUNAS (COM ESPAÇO!)
# =========================
ordem_correta = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE ",
    "ALLERGY ",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN"
]

# =========================
# BOTÃO
# =========================
if st.button("🔍 Analisar"):

    if nome.strip() == "":
        st.warning("Digite o nome do paciente")
    else:
        dados = pd.DataFrame([[ 
            gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol,
            coughing, shortness, swallowing, chest_pain
        ]], columns=ordem_correta)

        dados = dados[ordem_correta]

        prob = modelo.predict_proba(dados)[0][1]
        resultado = modelo.predict(dados)[0]

        # Texto final
        status = "Possível câncer" if resultado == 1 else "Sem indícios"

        # Salvar no histórico
        st.session_state.historico.append({
            "Nome": nome,
            "Risco (%)": f"{prob*100:.1f}",
            "Resultado": status
        })

        # Mostrar resultado
        st.subheader("Resultado")
        st.progress(float(prob))

        if resultado == 1:
            st.error(f"{nome}: {status}")
        else:
            st.success(f"{nome}: {status}")

# =========================
# HISTÓRICO
# =========================
st.divider()
st.subheader("📋 Histórico de Pacientes")

if len(st.session_state.historico) > 0:

    df_hist = pd.DataFrame(st.session_state.historico)

    st.dataframe(df_hist, use_container_width=True)

    # Botão limpar
    if st.button("🗑️ Limpar histórico"):
        st.session_state.historico = []
        st.rerun()

else:
    st.info("Nenhum paciente analisado ainda.")