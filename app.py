import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Configuración de página para que se vea bien en móviles (layout wide)
st.set_page_config(page_title="Estratega Pro v2.0", page_icon="🎯", layout="wide")

# --- 1. CONFIGURACIÓN Y ACTUALIZACIÓN ---
LIGAS_CONFIG = {
    "España 🇪🇸": {"Primera": "SP1", "Segunda": "SP2"},
    "Inglaterra 🏴󠁧󠁢󠁥󠁮󠁧󠁿": {"Premier League": "E0", "Championship": "E1"},
    "Italia 🇮🇹": {"Serie A": "I1"},
    "Alemania 🇩🇪": {"Bundesliga": "D1"},
    "Francia 🇫🇷": {"Ligue 1": "F1"},
    "Portugal 🇵🇹": {"Primeira Liga": "P1"}
}

def actualizar_datos_csv():
    temporadas = ['2526', '2425', '2324']
    lista_dfs = []
    with st.spinner('Actualizando Big Data...'):
        for pais, divisiones in LIGAS_CONFIG.items():
            for nombre_div, codigo_div in divisiones.items():
                for t in temporadas:
                    url = f"https://www.football-data.co.uk/mmz4281/{t}/{codigo_div}.csv"
                    try:
                        df_temp = pd.read_csv(url)
                        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY']
                        cols_actuales = [c for c in cols if c in df_temp.columns]
                        df_temp = df_temp[cols_actuales].dropna()
                        lista_dfs.append(df_temp)
                    except: continue
        if lista_dfs:
            df_final = pd.concat(lista_dfs, ignore_index=True)
            df_final['HomeTeam'] = df_final['HomeTeam'].str.strip()
            df_final['AwayTeam'] = df_final['AwayTeam'].str.strip()
            df_final.to_csv("datos_historicos.csv", index=False)
            return True
    return False

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("datos_historicos.csv")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except: return pd.DataFrame()

# --- 2. LÓGICA DE CÁLCULO ---
def obtener_stats(equipo, df):
    d = df[(df['HomeTeam'] == equipo) | (df['AwayTeam'] == equipo)].sort_values('Date').tail(30)
    if d.empty: return None
    g_f, c, f, t = [], [], [], []
    racha = 0
    for _, p in d.tail(5).iterrows():
        if p['HomeTeam'] == equipo:
            if p['FTR'] == 'H': racha += 3
            elif p['FTR'] == 'D': racha += 1
        else:
            if p['FTR'] == 'A': racha += 3
            elif p['FTR'] == 'D': racha += 1
    for _, p in d.iterrows():
        g_f.append(p['FTHG'] if p['HomeTeam'] == equipo else p['FTAG'])
        c.append(p['HC'] if p['HomeTeam'] == equipo else p['AC'])
        f.append(p['HF'] if p['HomeTeam'] == equipo else p['AF'])
        t.append(p['HY'] if p['HomeTeam'] == equipo else p['AY'])
    
    return {
        "goles": np.mean(g_f), 
        "corners": np.mean(c), 
        "faltas": np.mean(f), 
        "tarjetas": np.mean(t), 
        "racha": racha, 
        "fiabilidad": np.std(g_f)
    }

# --- 3. INTERFAZ ---
st.sidebar.title("🛠️ Panel de Control")
if st.sidebar.button("🔄 ACTUALIZAR BIG DATA"):
    if actualizar_datos_csv():
        st.cache_data.clear()
        st.sidebar.success("¡Base actualizada!")

df_total = cargar_datos()

if not df_total.empty:
    todos_los_equipos = sorted(df_total['HomeTeam'].unique())
    
    st.header("📊 Análisis de Élite")
    
    # SECCIÓN SELECCIÓN
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        loc = st.selectbox("🏠 Equipo Local", todos_los_equipos, index=todos_los_equipos.index("Real Madrid") if "Real Madrid" in todos_los_equipos else 0)
        b_atq_l = st.checkbox("Baja: Goleador (L)")
        b_def_l = st.checkbox("Baja: Defensa (L)")
        
    with col_sel2:
        vis = st.selectbox("🚀 Equipo Visitante", todos_los_equipos, index=todos_los_equipos.index("Barcelona") if "Barcelona" in todos_los_equipos else 1)
        b_atq_v = st.checkbox("Baja: Goleador (V)")
        b_def_v = st.checkbox("Baja: Defensa (V)")

    if st.button("🔍 EJECUTAR PREDICCIÓN COMPLETA", use_container_width=True):
        s_l, s_v = obtener_stats(loc, df_total), obtener_stats(vis, df_total)
        
        if s_l and s_v:
            # Ajustes Poisson
            atq_l = s_l['goles'] * (0.8 if b_atq_l else 1.0) * (1.15 if b_def_v else 1.0)
            atq_v = s_v['goles'] * (0.8 if b_atq_v else 1.0) * (1.15 if b_def_l else 1.0)
            exp_l, exp_v = atq_l * (1 + (s_l['racha']/30)) * 1.1, atq_v * (1 + (s_v['racha']/30))
            
            prob_l = [poisson.pmf(i, exp_l) for i in range(7)]
            prob_v = [poisson.pmf(i, exp_v) for i in range(7)]
            matriz = np.outer(prob_l, prob_v)
            res = np.unravel_index(np.argmax(matriz), matriz.shape)
            
            # Cálculo Probabilidad Marcar
            p_marcar_l = (1 - poisson.pmf(0, exp_l)) * 100
            p_marcar_v = (1 - poisson.pmf(0, exp_v)) * 100

            st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>Marcador Sugerido: {res[0]} - {res[1]}</h1>", unsafe_allow_html=True)
            
            # Fiabilidad
            st.divider()
            f1, f2 = st.columns(2)
            with f1: st.metric(f"Fiabilidad {loc}", "✅ ALTA" if s_l['fiabilidad'] < 1.0 else "⚠️ BAJA")
            with f2: st.metric(f"Fiabilidad {vis}", "✅ ALTA" if s_v['fiabilidad'] < 1.0 else "⚠️ BAJA")

            # H2H Vital
            st.subheader("🔙 Historial H2H (Últimos enfrentamientos)")
            h2h = df_total[((df_total['HomeTeam'] == loc) & (df_total['AwayTeam'] == vis)) | 
                           ((df_total['HomeTeam'] == vis) & (df_total['AwayTeam'] == loc))].sort_values('Date', ascending=False).head(10)
            if not h2h.empty:
                for _, f in h2h.iterrows():
                    st.write(f"📅 **{f['Date'].strftime('%d/%m/%Y')}** | {f['HomeTeam']} **{int(f['FTHG'])} - {int(f['FTAG'])}** {f['AwayTeam']} | 🚩 {int(f['HC']+f['AC'])} | 🟨 {int(f['HY']+f['AY'])} | ⚖️ {int(f['HF']+f['AF'])}")
            
            # Métricas Extra
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            p1, pX, p2 = np.sum(np.tril(matriz, -1)), np.diag(matriz).sum(), np.sum(np.triu(matriz, 1))
            with m1: st.write("**🛡️ Doble Op.**"); st.info(f"1X: {(p1+pX)*100:.1f}%\nX2: {(pX+p2)*100:.1f}%")
            with m2: st.write("**🎯 Marca +0.5**"); st.success(f"{loc}: {p_marcar_l:.1f}%\n{vis}: {p_marcar_v:.1f}%")
            with m3: st.write("**🚩 Corners/Tarj.**"); st.info(f"C: {round(s_l['corners']+s_v['corners'],1)}\nT: {round(s_l['tarjetas']+s_v['tarjetas'],1)}")
            with m4: st.write("**⚽ Goles**"); p_over = (1-(matriz[0,0]+matriz[0,1]+matriz[0,2]+matriz[1,0]+matriz[1,1]+matriz[2,0]))*100; st.info(f"Over 2.5: {p_over:.1f}%")

    # --- COMBINADA ---
    st.divider()
    st.header("🔥 Combinada Real-Time (Hoy)")
    if st.button("🚀 ESCANEAR JORNADA DE HOY"):
        with st.spinner('Leyendo marcadores...'):
            try:
                r = requests.get("https://www.resultados-futbol.com/hoy", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                soup = BeautifulSoup(r.text, 'html.parser')
                partidos = []
                for p in soup.find_all('tr', class_='vevent'):
                    l_w, v_w = p.find('td', class_='equipo1').text.strip(), p.find('td', class_='equipo2').text.strip()
                    partidos.append((l_w, v_w))
                
                combinada = []
                for l_w, v_w in partidos:
                    l_c = next((x for x in todos_los_equipos if l_w.lower() in x.lower() or x.lower() in l_w.lower()), None)
                    v_c = next((x for x in todos_los_equipos if v_w.lower() in x.lower() or x.lower() in v_w.lower()), None)
                    if l_c and v_c:
                        st_l = obtener_stats(l_c, df_total)
                        if st_l and st_l['racha'] >= 11 and st_l['fiabilidad'] < 1.0:
                            combinada.append(f"✅ **{l_c}** gana/empata vs {v_c}")
                        elif st_l and st_l['goles'] > 2.2:
                            combinada.append(f"⚽ **{l_c} vs {v_c}**: Over 1.5")
                    if len(combinada) >= 3: break
                
                if combinada:
                    st.success("📝 Propuesta para hoy:")
                    for c in combinada: st.write(c)
                else: st.warning("No hay 'chollos' estadísticos hoy.")
            except: st.error("Fallo de conexión.")

else: st.warning("Pulsa 'ACTUALIZAR BIG DATA' en el menú lateral.")