import time
import streamlit as st
import pyalgae_ai as AI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from hydroeval import evaluator, nse, kge

st.title("Sentinel-2 위성영상과 AI분석을 통한 녹조 분석 Web App")

st.subheader("(STEP-1) 위성관측자료와 녹조관측자료를 입력하세요")
csv_file = st.file_uploader("Select Your Local Observation CSV file")
if csv_file is not None:
    df = pd.read_csv(csv_file)
else:
    st.stop()
st.write(df.head())

st.subheader("(STEP-2) AI학습을 위해 입력자료로 활용할 위성자료와 답으로 활용할 녹조관측자료를 분리하세요")
divider1 = st.number_input(label="녹조관측자료의 첫번째 열",
                           min_value=0, max_value=len(list(df)), value=1)
divider2 = st.number_input(label="위성관측자료의 첫번째 열",
                           min_value=1, max_value=len(list(df)), value=1)

input = df.columns[divider2:len(list(df))]
label = df.columns[divider1:divider2]
input_sentinel = df[input]
label_algae = df[label]

if divider1 and divider2 is not None:
    st.write("AI학습을 위한 위성영상자료")
    st.write(input_sentinel)
    st.write("AI학습을 위한 녹조관측자료")
    st.write(label_algae)
else:
    st.stop()

st.subheader("(STEP-3) AI학습을 시행할 분광특성밴드 조합을 선택하세요")
select_columns1 = st.selectbox('위성영상의 밴드조합', options=[['B1 B2 B3'],
                                                      ['B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 AT CLOUD']])
select_columns2 = st.selectbox('위성영상의 밴드조합', options=[['B1 B2'],
                                                      ['B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12']])
select_columns = [select_columns1, select_columns2]

st.subheader("(STEP-4) AI학습을 모델, Training 크기 등을 선택하고 모델을 실행해 주세요")
if select_columns is not None:
    with st.form('user_inputs'):
        model = st.selectbox('AI적용 모델선택', options=[["RF"], ["GBR"], ["XGB"]])
        trainSize_rate = st.number_input(
            label="Training 데이터의 비율 (일반적으로 0.8 지정)", min_value=0.0, max_value=1.0, step=.1)
        n_estimators = st.number_input(
            label="분석할 가지의 갯수 지정", min_value=0, max_value=2000, step=100)
        st.form_submit_button()

        results = AI.algae_monitor(input_sentinel, label_algae, select_columns,
                                   model, trainSize_rate, n_estimators, random_state=42)
else:
    st.stop()
# st.write(results[1][0])
st.write("성공적으로 위성영상을 활용한 녹조 분석이 시행되었습니다.")


st.subheader("(STEP-5) 학습된 모델을 평가해주세요")

select_metrics = st.selectbox('Metrics 선택', options=[["NSE"], ["MSE"], [
                              "MAE"], ["RMSE"], ["R2"], ["KGE"]])

total_result = []
for i, value in enumerate(results):
    score_train = evaluator(nse, value[2], value[5], axis=1)
    score_test = evaluator(nse, value[4], value[6], axis=1)
    #score_train, score_test = AI.performance_test(select_metrics, value)
    f_result = '"{}" and "{}"의 결과: score_train={}, score_test={}'.format(
               ' '.join(list(value[1])), value[2].name, score_train, score_test)
    total_result.append(f_result)
    # st.write('"{}" and "{}"의 결과: score_train={}, score_test={}'.format(
    #    ' '.join(list(results[i][1])), results[i][2].name, score_train, score_test))
st.write(total_result)
