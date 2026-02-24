import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# 1. CONFIGURACIÓN E INTERFAZ
st.set_page_config(page_title="Pronósticos de fútbol profesional", layout="wide")

# Estilos de colores
COLOR_VERDE = "color: #2ecc71; font-weight: bold; font-size: 26px;"
COLOR_NORMAL = "font-size: 26px; font-weight: bold;"

# 2. FUNCIONES DE DATOS
def actualizar_csv():
    temporadas = ['2526', '2425', '2324']
    ligas = ["SP1", "SP2", "E0", "E1", "I1", "D1", "F1", "P1"]
    lista_dfs = []
    progreso = st.progress(0)
    for i, t in enumerate(temporadas):
        for cod in ligas:
            url = f"https://www.football-data.co.uk/mmz4281/{t}/{cod}.csv"
            try:
                df_temp = pd.read_csv(url)
                cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
                existentes = [c for c in cols if c in df_temp.columns]
                lista_dfs.append(df_temp[existentes])
            except: continue
        progreso.progress((i + 1) / len(temporadas))
    if lista_dfs:
        pd.concat(lista_dfs, ignore_index=True).to_csv("datos_historicos.csv", index=False)
        return True
    return False

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("datos_historicos.csv")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['HomeTeam'] = df['HomeTeam'].str.strip()
        df['AwayTeam'] = df['AwayTeam'].str.strip()
        return df
    except: return pd.DataFrame()

# 3. CUERPO DE LA APP
st.header("📊 Pronósticos de fútbol profesional")

if st.button("Actualizar datos"):
    if actualizar_csv():
        st.cache_data.clear()
        st.rerun()

df_total = cargar_datos()

if not df_total.empty:
    equipos = sorted(df_total['HomeTeam'].unique())
    col_l, col_v = st.columns(2)
    with col_l: loc = st.selectbox("🏠 Local", equipos, index=0)
    with col_v: vis = st.selectbox("🚀 Visitante", equipos, index=1)

    # Filtrado de datos (últimos 20 partidos)
    d_l = df_total[(df_total['HomeTeam'] == loc) | (df_total['AwayTeam'] == loc)].tail(20)
    d_v = df_total[(df_total['HomeTeam'] == vis) | (df_total['AwayTeam'] == vis)].tail(20)

    if not d_l.empty and not d_v.empty:
        # --- CÁLCULO DE MEDIAS SEGURO ---
        def get_mean(df, col, default=0.0):
            if col in df.columns:
                val = df[col].mean()
                return val if pd.notnull(val) else default
            return default

        m_l = d_l[d_l['HomeTeam'] == loc]['FTHG'].mean() if not d_l[d_l['HomeTeam'] == loc].empty else d_l['FTHG'].mean()
        m_v = d_v[d_v['AwayTeam'] == vis]['FTAG'].mean() if not d_v[d_v['AwayTeam'] == vis].empty else d_v['FTAG'].mean()
        
        # Probabilidades de Goles (Poisson)
        mu_total = m_l + m_v
        p_under_2 = poisson.cdf(2, mu_total) 
        prob_over_25 = (1 - p_under_2) * 100

        # Matriz para Victoria/Empate
        p_l_list = [poisson.pmf(i, m_l) for i in range(7)]
        p_v_list = [poisson.pmf(i, m_v) for i in range(7)]
        matriz = np.outer(p_l_list, p_v_list)
        
        p_win_l = np.sum(np.tril(matriz, -1))
        p_empate = np.diag(matriz).sum()
        p_win_v = np.sum(np.triu(matriz, 1))

        # --- VISUALIZACIÓN ---
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("🎯 Marcador Sugerido")
            m_sug = np.unravel_index(np.argmax(matriz), matriz.shape)
            st.markdown(f"<h1 style='color:#FF4B4B;'>{m_sug[0]} - {m_sug[1]}</h1>", unsafe_allow_html=True)
            st.write(f"Victoria Local: {p_win_l*100:.1f}%")
            st.write(f"Empate: {p_empate*100:.1f}%")
            st.write(f"Victoria Visitante: {p_win_v*100:.1f}%")

        with c2:
            st.subheader("🛡️ Doble Oportunidad")
            st.info(f"**1X (Local o Empate):** {(p_win_l + p_empate)*100:.1f}%")
            st.info(f"**X2 (Visitante o Empate):** {(p_win_v + p_empate)*100:.1f}%")

        with c3:
            st.subheader("⚽ Probabilidad Goles")
            color = COLOR_VERDE if prob_over_25 > 60 else COLOR_NORMAL
            st.markdown(f"**Over 2.5 Goles:**")
            st.markdown(f"<p style='{color}'>{prob_over_25:.1f}%</p>", unsafe_allow_html=True)

        # Fila 2: Corners, Tarjetas, Faltas
        st.divider()
        st.subheader("📈 Expectativas Estadísticas")
        e1, e2, e3 = st.columns(3)
        
        c_tot = get_mean(d_l, 'HC', 4.5) + get_mean(d_v, 'AC', 4.0)
        t_tot = (get_mean(d_l, 'HY', 2.0) + get_mean(d_l, 'HR', 0.1)) + (get_mean(d_v, 'AY', 2.0) + get_mean(d_v, 'AR', 0.1))
        f_tot = get_mean(d_l, 'HF', 11.5) + get_mean(d_v, 'AF', 11.5)
        
        e1.metric("Corners", round(c_tot, 1))
        e2.metric("Tarjetas", round(t_tot, 1))
        e3.metric("Faltas", round(f_tot, 1))

        # Historial H2H
        st.divider()
        st.subheader("🔙 Historial H2H Detallado")
        st.caption("Goles | Corners | Tarjetas | Faltas")
        h2h = df_total[((df_total['HomeTeam'] == loc) & (df_total['AwayTeam'] == vis)) | 
                       ((df_total['HomeTeam'] == vis) & (df_total['AwayTeam'] == loc))].sort_values('Date', ascending=False).head(8)
        
        if not h2h.empty:
            for _, r in h2h.iterrows():
                co = int(r.get('HC',0)+r.get('AC',0))
                ta = int(r.get('HY',0)+r.get('AY',0)+r.get('HR',0)+r.get('AR',0))
                fa = int(r.get('HF',0)+r.get('AF',0))
                st.write(f"📅 {r['Date'].strftime('%d/%m/%Y')} | **{r['HomeTeam']} {int(r['FTHG'])} - {int(r['FTAG'])} {r['AwayTeam']}** | 🚩 {co} | 🟨 {ta} | ⚖️ {fa}")
        else:
            st.info("No hay enfrentamientos directos previos registrados.")

else: st.warning("Pulsa 'Actualizar datos' para cargar la base histórica.")
