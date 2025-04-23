import streamlit as st
import pandas as pd
import joblib
from recommend import recommend_md

# Загрузка моделей
model_output_cat = joblib.load("models/model_output_cat.pkl")
model_output_xgb = joblib.load("models/model_output_xgb.pkl")
model_cp2o5_cat = joblib.load("models/model_cp2o5_cat.pkl")
model_cp2o5_xgb = joblib.load("models/model_cp2o5_xgb.pkl")
model_cmgo_rf = joblib.load("models/model_cmgo_rf.pkl")
model_cmgo_cat = joblib.load("models/model_cmgo_cat.pkl")

st.title("📊 Прогноз результатов РСУ на фос. руде с рекомендациями")

fraction = st.selectbox("Фракция", ['20-40', '40-80', '80-130'])
f_p2o5 = st.number_input("F_P2O5_%", value=24.5)
f_mgo = st.number_input("F_MgO_%", value=4.1)
feed = st.number_input("Подача (т)", value=250)
priority = st.selectbox("Приоритет", ['Output', 'P2O5', 'MgO'])

if st.button("🔍 Рассчитать рекомендации"):
    top5, full = recommend_md(
        fraction_str=fraction,
        f_p2o5=f_p2o5,
        f_mgo=f_mgo,
        feed=feed,
        prioritet=priority,
        model_output_cat=model_output_cat,
        model_output_xgb=model_output_xgb,
        model_cp2o5_cat=model_cp2o5_cat,
        model_cp2o5_xgb=model_cp2o5_xgb,
        model_cmgo_rf=model_cmgo_rf,
        model_cmgo_cat=model_cmgo_cat
    )

    st.subheader("✅ Топ-5 рекомендаций:")
    st.dataframe(top5.round(2))

    st.subheader("📋 Полная таблица:")
    st.dataframe(full.round(2))
