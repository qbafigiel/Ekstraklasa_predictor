import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from scraper import pobierz_statystyki_druzyn

st.set_page_config(
    page_title="Ekstraklasa Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Ekstraklasa Predictor")
st.subheader("System predykcji meczów PKO BP Ekstraklasy")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Sezon", value="2025/26")
with col2:
    st.metric(label="Drużyny", value="18")
with col3:
    st.metric(label="Status", value="🟢 Działa")

st.divider()

st.header("📊 Statystyki drużynowe")

with st.spinner("Pobieranie danych z ekstraklasa.org..."):
    df = pobierz_statystyki_druzyn()

if df is not None:
    st.success(f"Pobrano dane dla {len(df)} drużyn")
    
    kolumny_nazwy = {
        'LP.': 'LP',
        'KLUB': 'Klub',
        'PKT': 'Pkt',
        'PUS': 'Mecze',
        'Z': 'W',
        'R': 'R', 
        'P': 'P',
        'POS': 'Posiadanie %',
        'GOL': 'Gole zdobyte',
        'SGL': 'Gole stracone',
        'XGL': 'xG',
        'XGM': 'xG/mecz',
        'STR': 'Strzały/mecz',
        'STC': 'Strzały celne/mecz',
        'FAU': 'Faule',
        'ŻK': 'ŻK',
        'CZK': 'CzK'
    }
    
    kolumny_pokazane = ['KLUB', 'PKT', 'PUS', 'Z', 'R', 'P', 
                        'GOL', 'SGL', 'XGL', 'XGM', 
                        'STR', 'STC', 'FAU', 'ŻK', 'CZK']
    
    kolumny_dostepne = [k for k in kolumny_pokazane if k in df.columns]
    df_show = df[kolumny_dostepne].copy()
    df_show = df_show.rename(columns=kolumny_nazwy)
    
    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True
    )
else:
    st.error("Nie udało się pobrać danych. Sprawdź połączenie z internetem.")