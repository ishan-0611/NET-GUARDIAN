import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics
import matplotlib.pyplot as plt


st.title('Net Guardian :shield: :globe_with_meridians:')
st.write('''#### A Machine Learning App for Network Anomaly Detection''')

st.sidebar.title('Parameter Selection :')
model = st.sidebar.selectbox('Select a Machine Learning Algorithm :',
                     ['Logistic Regression',
                      'Decision Tree Classifier',
                      'Random Forest Classifier',
                      'Naive Bayes Classifier'])

size = st.sidebar.selectbox('Choose size for DataSet :', [0.10, 0.25, 0.30, 0.5, 1.0])

st.sidebar.subheader('Details :')
link = st.sidebar.link_button(label = 'DataSet', url = 'https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data?select=UNSW_NB15_training-set.csv')

st.sidebar.subheader('Research Papers :')
st.sidebar.link_button(label = 'Paper 1', url = 'https://drive.google.com/file/d/1XwvH6DyBif2o53CtLHoo33xaumjdJfK3/view?usp=share_link')
st.sidebar.link_button(label = 'Paper 2', url = 'https://drive.google.com/file/d/1Ti6FB5fSDx8GhPIZ8vyBHUxNvmurgO3A/view?usp=share_link')
st.sidebar.link_button(label = 'Paper 3', url = 'https://drive.google.com/file/d/1T3IE3xE8I7VouWwF2Bpu40F_XpfiCcX2/view?usp=share_link')


def make_prediction(model, size):

    if model == 'Logistic Regression':
        ml_model = joblib.load('Log_Reg.joblib')
    elif model == 'Decision Tree Classifier':
        ml_model = joblib.load('Dec_Tree.joblib')
    elif model == 'Random Forest Classifier':
        ml_model = joblib.load('Random_Forest.joblib')
    elif model == 'Naive Bayes Classifier':
        ml_model = joblib.load('Naive_Bayes.joblib')


    test = pd.read_csv('UNSW_NB15_testing-set.csv')
    test = test.sample(frac = size, random_state = 42)

    object_cols = test.select_dtypes(include = 'object').columns
    test.drop(columns = object_cols, inplace = True)

    x = test.drop('label', axis = 1)
    y = test['label']
    y_preds = ml_model.predict(x)
    acc = ml_model.score(x, y)
    prec_score = metrics.precision_score(y_pred = y_preds, y_true = y)
    recall_score = metrics.recall_score(y_pred = y_preds, y_true = y)
    f1_score = metrics.f1_score(y_pred = y_preds, y_true = y)

    return acc, prec_score, recall_score, f1_score


def compare_models():

    test = pd.read_csv('UNSW_NB15_testing-set.csv')
    test = test.sample(frac = size, random_state = 42)

    object_cols = test.select_dtypes(include = 'object').columns
    test.drop(columns = object_cols, inplace = True)

    x = test.drop('label', axis = 1)
    y = test['label']

    y_pred_dict = {}
    models = {
    'Logistic Regression': joblib.load('Log_Reg.joblib'),
    'Decision Tree':joblib.load('Dec_Tree.joblib'),
    'Random Forest': joblib.load('Random_Forest.joblib'),
    'Naive Bayes': joblib.load('Naive_Bayes.joblib')}

    for model_name, model in models.items():
        y_pred = model.predict(x)
        y_pred_dict[model_name] = y_pred
    
    evaluation_results = []
    for model_name, y_pred in y_pred_dict.items():
        accuracy = metrics.accuracy_score(y, y_pred)
        precision = metrics.precision_score(y, y_pred)
        recall = metrics.recall_score(y, y_pred)
        f1 = metrics.f1_score(y, y_pred)

        evaluation_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1})

    accuracy_results = []
    for model_name, y_pred in y_pred_dict.items():
        accuracy = metrics.accuracy_score(y, y_pred) * 100
        accuracy_results.append({
            'Model': model_name,
            'Accuracy': f"{accuracy:.2f}%"})
        
    df_results = pd.DataFrame(evaluation_results)
    st.dataframe(df_results)
    df_accuracy = pd.DataFrame(accuracy_results)
    df_accuracy_sorted = df_accuracy.sort_values(by = 'Accuracy', ascending = False)

    st.subheader('Accuracy Sorted :')
    st.dataframe(df_accuracy_sorted)
    st.subheader('Graphical Analysis :')
    st.bar_chart(df_accuracy_sorted.set_index('Model'))    



st.write('''---''')
st.write('''##### Selected Model :   {}'''.format(model))
st.write('''##### Selected Dataset Size :  {} {}'''.format(size * 100, '%'))


with st.container():
    tab1, tab2 = st.tabs(['Run Tests', 'Compare'])

    with tab1:
        ok = st.button(label = 'Run Selected Model')
        if ok:
            with st.spinner('Running the Model ...'):
                time.sleep(3)

            accuracy, prec_score, recall_score, f1_score = make_prediction(model, size)

            data = {'Precision Score' : [f"{prec_score * 100:.2f} %"],
                    'Recall Score' : [f"{recall_score * 100:.2f} %"],
                    'F1 Score' : [f"{f1_score * 100:.2f} %"]}
            
            st.write('''#### Accuracy : {} %'''.format(round(accuracy * 100, 2)))
            st.dataframe(data)

            
    with tab2:
        ok1 = st.button(label = 'Compare All Models')
        if ok1:
            with st.spinner('Generating Insights from Models .... '):
                time.sleep(3)

            st.subheader('Analysis of all Models : ')
            compare_models()
