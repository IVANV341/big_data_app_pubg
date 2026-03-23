import streamlit as st
import pickle
import numpy as np

# ── Configuracion ───────────────────────────────────────
st.set_page_config(
    page_title="PUBG Placement Predictor",
    page_icon="🎮",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    with open('modelo_gbt.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return modelo, le

modelo, le = cargar_modelo()

# ── Estilos ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo oscuro general */
    .stApp {
        background: linear-gradient(160deg, #0a0e1a 0%, #0d1520 50%, #0a1628 100%);
    }

    /* Header principal */
    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #F5A623, #ff6b35, #F5A623);
    }
    .hero-titulo {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #F5A623;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin: 0;
        text-shadow: 0 0 30px rgba(245,166,35,0.4);
    }
    .hero-sub {
        font-size: 1rem;
        color: #7a9cc4;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(245,166,35,0.15);
        border: 1px solid rgba(245,166,35,0.4);
        color: #F5A623;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin-top: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Tarjetas de seccion */
    .card {
        background: rgba(13, 27, 42, 0.8);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .card-titulo {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #F5A623;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-left: 3px solid #F5A623;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }

    /* Badge de importancia */
    .badge-alta {
        display: inline-block;
        background: rgba(245,166,35,0.2);
        border: 1px solid rgba(245,166,35,0.5);
        color: #F5A623;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        margin-bottom: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .badge-baja {
        display: inline-block;
        background: rgba(100,130,180,0.15);
        border: 1px solid rgba(100,130,180,0.3);
        color: #6496c8;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        margin-bottom: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .hint {
        font-size: 0.75rem;
        color: #4a6a8a;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    /* Sliders y inputs */
    .stSlider > div > div > div > div {
        background: #F5A623 !important;
    }
    .stSlider > div > div > div {
        background: rgba(30, 58, 95, 0.5) !important;
    }

    /* Boton */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #F5A623, #ff8c00);
        color: #0a0e1a;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 0.85rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 20px rgba(245,166,35,0.3);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffc040, #F5A623);
        box-shadow: 0 6px 25px rgba(245,166,35,0.5);
        transform: translateY(-1px);
    }

    /* Resultado */
    .resultado-winner {
        background: linear-gradient(135deg, #1a2a0a, #2a3a0a);
        border: 2px solid #F5A623;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 0 40px rgba(245,166,35,0.3);
    }
    .resultado-excelente {
        background: linear-gradient(135deg, #0a2a0a, #0d3a0d);
        border: 1px solid #4CAF50;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .resultado-bueno {
        background: linear-gradient(135deg, #0a1a3a, #0d2a4a);
        border: 1px solid #2196F3;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .resultado-medio {
        background: linear-gradient(135deg, #2a1a0a, #3a2a0a);
        border: 1px solid #F5A623;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .resultado-malo {
        background: linear-gradient(135deg, #2a0a0a, #3a0d0d);
        border: 1px solid #F44336;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .puesto-num {
        font-family: 'Rajdhani', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    .puesto-label {
        font-size: 1rem;
        color: #7a9cc4;
        margin-top: 0.25rem;
        letter-spacing: 1px;
    }
    .chicken-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #F5A623;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Consejos */
    .consejo {
        background: rgba(13, 27, 42, 0.9);
        border-left: 3px solid #F5A623;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.92rem;
        color: #c8d8e8;
        line-height: 1.5;
    }

    /* Metricas */
    [data-testid="stMetric"] {
        background: rgba(13, 27, 42, 0.8);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.75rem;
    }
    [data-testid="stMetricLabel"] { color: #6496c8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #F5A623 !important; font-weight: 700 !important; }

    /* Radio y selectbox */
    .stRadio label { color: #c8d8e8 !important; }
    .stSelectbox label { color: #c8d8e8 !important; }
    label { color: #c8d8e8 !important; }
    p { color: #c8d8e8; }

    /* Divider */
    hr { border-color: #1e3a5f !important; }

    /* Barra de progreso de importancia */
    .barra-container {
        background: rgba(30, 58, 95, 0.3);
        border-radius: 4px;
        height: 6px;
        margin: 4px 0 12px 0;
        overflow: hidden;
    }
    .barra-fill-alta {
        background: linear-gradient(90deg, #F5A623, #ff8c00);
        height: 100%;
        border-radius: 4px;
    }
    .barra-fill-baja {
        background: linear-gradient(90deg, #2196F3, #1565C0);
        height: 100%;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── HERO ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-titulo">PUBG Predictor</div>
    <div class="hero-sub">Descubre en que puesto habrias terminado</div>
    <div class="hero-badge">Powered by Gradient Boosting · MAE 0.07</div>
</div>
""", unsafe_allow_html=True)

# ── Selector de modo ────────────────────────────────────
modo_ingreso = st.radio(
    "Modo de ingreso:",
    ["Sliders — arrastra para ajustar", "Manual — escribe el valor exacto"],
    horizontal=True
)
usar_slider = "Slider" in modo_ingreso

st.markdown("<br>", unsafe_allow_html=True)

# ── Funcion input ───────────────────────────────────────
def input_stat(label, min_val, max_val, default, step, ayuda, pct_barra, es_alta, key):
    color_badge = "badge-alta" if es_alta else "badge-baja"
    color_barra = "barra-fill-alta" if es_alta else "barra-fill-baja"
    nivel = "ALTA IMPORTANCIA" if es_alta else "importancia secundaria"
    st.markdown(f'<span class="{color_badge}">{nivel}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="barra-container"><div class="{color_barra}" style="width:{pct_barra}%"></div></div>', unsafe_allow_html=True)
    if usar_slider:
        val = st.slider(label, min_value=min_val, max_value=max_val,
                        value=default, step=step, key=key)
    else:
        val = st.number_input(label, min_value=float(min_val), max_value=float(max_val),
                              value=float(default), step=float(step), key=key)
        val = int(val) if step == 1 else float(val)
    st.markdown(f'<p class="hint">{ayuda}</p>', unsafe_allow_html=True)
    return val

# ── SECCION 1: Variables clave ──────────────────────────
st.markdown('<div class="card"><div class="card-titulo">Variables que mas importan — 87% del modelo</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    walkDistance = input_stat(
        "Metros caminados a pie",
        0, 9000, 1143, 50,
        "Promedio: 1,143 m | Top players: 5,000+ m",
        69, True, "walk"
    )
with col2:
    killPlace = input_stat(
        "Ranking de kills (1=mas kills · 100=menos kills)",
        1, 100, 47, 1,
        "Promedio: 47 | Top players: entre 10 y 30",
        18, True, "killplace"
    )
st.markdown('</div>', unsafe_allow_html=True)

# ── SECCION 2: Variables secundarias ───────────────────
st.markdown('<div class="card"><div class="card-titulo">Variables secundarias — 13% restante</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    duracion_min = input_stat(
        "Duracion de la partida (minutos)",
        6, 36, 26, 1,
        "Promedio: 26 min | Top players: 30+ min",
        21, False, "duracion"
    )
    matchDuration = int(duracion_min) * 60
with col2:
    damageDealt = input_stat(
        "Dano total causado",
        0, 4000, 130, 10,
        "Promedio: 130 | Top players: 200+",
        9, False, "damage"
    )

col1, col2, col3 = st.columns(3)
with col1:
    kills = input_stat("Enemigos eliminados", 0, 46, 1, 1,
                       "Promedio: 1", 5, False, "kills")
with col2:
    boosts = input_stat("Items de energia usados", 0, 14, 1, 1,
                        "Promedio: 1 | Top: 5+", 5, False, "boosts")
with col3:
    heals = input_stat("Items de curacion usados", 0, 48, 1, 1,
                       "Promedio: 1", 4, False, "heals")

col1, col2 = st.columns(2)
with col1:
    weaponsAcquired = input_stat("Armas recogidas", 0, 37, 4, 1,
                                 "Promedio: 4", 4, False, "weapons")
with col2:
    rideDistance = input_stat("Metros en vehiculo", 0, 14000, 579, 100,
                              "Promedio: 579 m", 3, False, "ride")

st.markdown('</div>', unsafe_allow_html=True)

# ── SECCION 3: Modo de juego ────────────────────────────
st.markdown('<div class="card"><div class="card-titulo">Modo de juego</div>', unsafe_allow_html=True)
modo = st.selectbox(
    "En que modo jugaste?",
    options=[
        "squad-fpp -- Squad primera persona (mas comun)",
        "duo-fpp   -- Duo primera persona",
        "solo-fpp  -- Solo primera persona",
        "squad     -- Squad tercera persona",
        "duo       -- Duo tercera persona",
        "solo      -- Solo tercera persona",
    ]
)
matchType = modo.split("--")[0].strip()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── BOTON ───────────────────────────────────────────────
if st.button("PREDECIR MI PUESTO"):

    known_classes = list(le.classes_)
    matchTypeIdx = le.transform([matchType])[0] if matchType in known_classes else le.transform(["squad-fpp"])[0]

    features = np.array([[
        0, boosts, damageDealt, 0, 0, heals,
        killPlace, 512, kills,
        1 if kills > 1 else 0,
        23.0, matchDuration, 45, 43, 883, 0,
        rideDistance, 0, 0, 0, 0,
        walkDistance, weaponsAcquired, 617, matchTypeIdx
    ]])

    winPlacePerc = float(np.clip(modelo.predict(features)[0], 0, 1))
    puesto = round((1 - winPlacePerc) * 99) + 1

    # ── Resultado ───────────────────────────────────────
    if puesto == 1:
        st.markdown("""
        <div class="resultado-winner">
            <div style="font-size:3rem">🏆</div>
            <div class="chicken-text">WINNER WINNER CHICKEN DINNER!</div>
            <div class="puesto-label" style="margin-top:0.5rem">LLEGASTE AL PUESTO #1 DE 100</div>
        </div>""", unsafe_allow_html=True)
        st.balloons()
    elif puesto <= 10:
        st.markdown(f"""
        <div class="resultado-excelente">
            <div style="font-size:2rem">🥇</div>
            <div class="puesto-num" style="color:#4CAF50">#{puesto}</div>
            <div class="puesto-label">de 100 jugadores — Resultado excelente!</div>
        </div>""", unsafe_allow_html=True)
    elif puesto <= 30:
        st.markdown(f"""
        <div class="resultado-bueno">
            <div style="font-size:2rem">🎯</div>
            <div class="puesto-num" style="color:#2196F3">#{puesto}</div>
            <div class="puesto-label">de 100 jugadores — Buen resultado</div>
        </div>""", unsafe_allow_html=True)
    elif puesto <= 60:
        st.markdown(f"""
        <div class="resultado-medio">
            <div style="font-size:2rem">⚔️</div>
            <div class="puesto-num" style="color:#F5A623">#{puesto}</div>
            <div class="puesto-label">de 100 jugadores — Puedes mejorar</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="resultado-malo">
            <div style="font-size:2rem">💀</div>
            <div class="puesto-num" style="color:#F44336">#{puesto}</div>
            <div class="puesto-label">de 100 jugadores — Eliminated early...</div>
        </div>""", unsafe_allow_html=True)

    # ── Consejos ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-titulo" style="border-left:3px solid #F5A623; padding-left:0.75rem;">Consejos para mejorar</div>', unsafe_allow_html=True)

    consejos = []
    if puesto <= 15:
        consejos.append("Partida excelente — tu combinacion de movimiento y kills es la clave")
        if walkDistance > 4000:
            consejos.append("Tu distancia caminada es tu mayor fortaleza — sigue moviendote asi")
        if killPlace <= 25:
            consejos.append("Buen ranking de kills — es lo segundo que mas pesa en el modelo (18%)")
    elif puesto <= 35:
        consejos.append("Buen resultado — con pequenos ajustes puedes entrar al top 10")
        if walkDistance < 4000:
            consejos.append("Camina mas — intenta superar los 4,000 metros por partida (69% del modelo)")
        if killPlace > 30:
            consejos.append("Mejora tu ranking de kills — intenta bajar de 30 eliminando mas enemigos")
    elif puesto <= 60:
        consejos.append("Resultado medio — enfocate en las 2 variables que mas pesan")
        if walkDistance < 3000:
            consejos.append("Camina mucho mas — walkDistance vale el 69% del modelo, es lo mas importante")
        if killPlace > 40:
            consejos.append("Tu ranking de kills esta muy alto — elimina mas enemigos para bajarlo")
        if matchDuration < 1500:
            consejos.append("Aguanta mas en partida — durar mas minutos mejora tu posicion")
    else:
        consejos.append("Necesitas cambiar tu estrategia — enfocate en estas 2 cosas:")
        consejos.append("Camina mas de 4,000 metros — es el factor mas importante con 69% de peso")
        consejos.append("Baja tu ranking de kills por debajo de 30 — vale el 18% de la prediccion")
        if matchDuration < 900:
            consejos.append("Estas muriendo muy rapido — evita confrontaciones al inicio de la partida")

    for c in consejos:
        st.markdown(f'<div class="consejo">{c}</div>', unsafe_allow_html=True)

    # ── Resumen stats ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-titulo" style="border-left:3px solid #F5A623; padding-left:0.75rem;">Resumen de tu partida</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kill Ranking", f"#{killPlace}", delta=f"{47 - killPlace:+} vs avg")
    col2.metric("Distancia", f"{walkDistance}m", delta=f"{walkDistance - 1143:+} vs avg")
    col3.metric("Duracion", f"{duracion_min} min", delta=f"{int(duracion_min) - 26:+} vs avg")
    col4.metric("Dano", f"{damageDealt}", delta=f"{damageDealt - 130:+} vs avg")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#2a4a6a; font-size:0.75rem;">Modelo: Gradient Boosting | Entrenado con datos reales de PUBG | MAE: 0.0708</p>', unsafe_allow_html=True)