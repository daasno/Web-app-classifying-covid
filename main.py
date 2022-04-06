import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Will covid kill your passion or not?")
    st.sidebar.markdown("Will covid kill your passion or not?")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('data_cov.csv')
        data.drop(index=data.index[0], axis=0, inplace=True)
        # label = LabelEncoder()
        # for col in data.columns:
        #     data[col] = label.fit_transform(data[col])
        return data

    # def init_data(df):
    #     y = df.result
    #     x = df.drop(columns=['result'])
    #     return x, y

    # @st.cache(persist=True)
    def split(df):
        y = df.result
        x = df.drop(columns=['result'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x, y, x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x, y, x_train, x_test, y_train, y_test = split(df)

    class_names = [1, 0]
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("Support Vector Machine (SVM)", "Naive Bayes"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization paramter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        cv = st.sidebar.selectbox("cross validation folds", (5, 10), key='cv')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine(SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            scores = cross_val_score(model, x, y, cv=cv)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Score: ", scores.mean())
            plot_metrics(metrics)

    if classifier == 'Naive Bayes':
        st.sidebar.subheader("Model Hyperparameters")
        st.sidebar.markdown("Model got no hyperparametres")
        cv = st.sidebar.selectbox("cross validation folds", (5, 10), key='cv')
        # C = st.sidebar.number_input("C (Regularization paramter)", 0.01, 10.0, step=0.01, key='C_LR')
        # max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Naive Bayes Results")
            model = GaussianNB()
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            scores = cross_val_score(model, x, y, cv=cv)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Score: ", scores.mean())
            st.write(f"Number of mislabeled points out of a total %d points : %d", (x_test.shape[0], (y_test != y_pred).sum()))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
