import streamlit as st
import pandas as pd
import pickle
header = st.container()
with header:
    st.title("Предсказательная диагностическая модель развития рака легкого")
    st.write("Вероятность развития рака легкого")

    data = st.file_uploader("Загрузите файл")
if data is not None:
    data = pd.read_excel(data)
    loaded_model = pickle.load(open('lung cancer RF model.pkl', 'rb'))
    st.write(data)
    a=loaded_model.predict_proba(data)
    b=float(a[:,1])*100
    st.write('вероятность развития рака легкого = ', round(b,2), '%')






