import streamlit as st
import pandas as pd
import os
from pathlib import Path

import shap
import matplotlib.pyplot as plt

from src.pipeline import load_artifacts, transform

MODEL_DIR = Path(__file__).resolve().parent / "models" / "pretrained"

st.set_page_config(
    page_title="Диагностика патологий печени",
    layout="wide",
)

st.title("Экспресс-диагностика патологий печени")
st.markdown("""
Данное приложение использует предобученную модель **XGBoost** для оценки риска наличия заболеваний печени 
(Гепатит C, Фиброз, Цирроз) на основе биохимического анализа крови.
""")
st.markdown("---")

@st.cache_resource
def load_ml_artifacts():
    return load_artifacts(MODEL_DIR)

try:
    model, robust_scaler, medians_dict, outlier_bounds = load_ml_artifacts()
except FileNotFoundError as e:
    st.error(f"Ошибка загрузки артефактов: не найден файл {e.filename}")
    st.info(f"Проверьте наличие четырёх файлов модели в каталоге {MODEL_DIR}")
    st.stop()

# Разделяем экран на две равные колонки
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Введите показатели пациента")

    with st.form("patient_data_form"):
        col1, col2 = st.columns(2)
        user_data = {}

        with col1:
            user_data['Age'] = st.number_input("Возраст (лет)", min_value=0, max_value=120, value=40, step=1)
            user_data['Sex'] = st.selectbox("Пол", options=["Мужской", "Женский"], index=0)

            user_data['ALB'] = st.number_input("Альбумин (ALB, г/л)", value=float(medians_dict['ALB']), step=0.1)
            user_data['ALP'] = st.number_input("Щелочная фосфатаза (ALP, Ед/л)", value=float(medians_dict['ALP']),
                                               step=0.1)
            user_data['ALT'] = st.number_input("Аланинаминотрансфераза (ALT, Ед/л)", value=float(medians_dict['ALT']),
                                               step=0.1)
            user_data['AST'] = st.number_input("Аспартатаминотрансфераза (AST, Ед/л)", value=float(medians_dict['AST']),
                                               step=0.1)

        with col2:
            user_data['BIL'] = st.number_input("Билирубин (BIL, мкмоль/л)", value=float(medians_dict['BIL']), step=0.1)
            user_data['CHE'] = st.number_input("Холинэстераза (CHE, кЕд/л)", value=float(medians_dict['CHE']), step=0.1)
            user_data['CHOL'] = st.number_input("Холестерин (CHOL, ммоль/л)", value=float(medians_dict['CHOL']),
                                                step=0.1)
            user_data['CREA'] = st.number_input("Креатинин (CREA, мкмоль/л)", value=float(medians_dict['CREA']),
                                                step=0.1)
            user_data['GGT'] = st.number_input("Гамма-глутамилтрансфераза (GGT, Ед/л)",
                                               value=float(medians_dict['GGT']), step=0.1)
            user_data['PROT'] = st.number_input("Общий белок (PROT, г/л)", value=float(medians_dict['PROT']), step=0.1)

        submitted = st.form_submit_button("Запустить диагностику")
    st.markdown("---")
    with st.expander("Как интерпретировать SHAP-график?"):
        st.markdown("""
            **Цветовая кодировка на графике-водопаде:**

        - <span style="color:#ff0051">**Красный цвет:** признак **повышает** риск патологии печени</span>
        - <span style="color:#008bfb">**Синий цвет:** признак **снижает** риск патологии печени</span>

            **Длина столбца** показывает силу влияния:
            - Чем длиннее столбец, тем больше вклад признака в решение модели
            - Столбцы направлены влево (снижают риск) или вправо (повышают риск)
            """, unsafe_allow_html=True)
with right_col:
    st.header("Анализ и интерпретация")

    if submitted:
        # предобработка данных
        df_input = pd.DataFrame([user_data])
        df_input['Sex'] = df_input['Sex'].map({"Мужской": 0, "Женский": 1})

        df_input_scaled = transform(df_input, robust_scaler, medians_dict, outlier_bounds)

        # расчет вероятностей моделью
        prob_healthy = model.predict_proba(df_input_scaled)[0][0]
        prob_disease = model.predict_proba(df_input_scaled)[0][1]

        # вывод метрик вероятности
        st.subheader("Результаты диагностики")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Вероятность нормы (Blood Donor)", value=f"{prob_healthy * 100:.1f}%")
        with col_res2:
            st.metric(label="Вероятность патологии", value=f"{prob_disease * 100:.1f}%")

        if prob_disease >= 0.5:
            st.error("Обнаружен высокий риск патологии печени!")
        else:
            st.success("Показатели находятся в пределах нормы.")

        st.markdown("---")
        st.subheader("Визуализация вклада признаков (SHAP Waterfall)")

        with st.spinner("Расчет вклада биомаркеров..."):
            explainer_shap = shap.TreeExplainer(model)
            shap_values_input = explainer_shap(df_input_scaled)

            shap_values_input.data = df_input.values

            fig, ax = plt.subplots(figsize=(10, 5))

            # передаем объект в waterfall
            shap.plots.waterfall(shap_values_input[0], show=False)

            plt.tight_layout()

            # выводим график в интерфейс Streamlit
            st.pyplot(fig)
            plt.close()  # очищаем оперативку от фигуры

    else:
        st.info(
            "Заполните медицинские показатели пациента слева и нажмите кнопку 'Запустить диагностику' для вывода результатов и построения графика SHAP.")
