import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my machine learning dashboard

This dashboard created by : [@yourname](https://www.linkedin.com/in/yourname/)
""")
add_selectitem = st.sidebar.selectbox("Want to open about?", (" ", "Iris species!"))
def iris():
    st.write("""
    This app predicts the **Iris Species**

    Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML.
    """)
    st.sidebar.header('User Input Features:')
   uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', 4.3,6.5,10.0)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', 2.0,3.3,5.0)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)', 1.0,4.5,9.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)', 0.1,1.4,5.0)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0])
            return features
input_df = user_input_features()
    img = Image.open("iris.JPG")
    st.image(img, width=500)
if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("model_iris.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)
        result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Memuat dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Pra-pemrosesan data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Memisahkan data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membangun model SVM
model = SVC()

# Definisi grid parameter untuk hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}

# Melakukan hyperparameter tuning menggunakan GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Mendapatkan model terbaik dan parameter terbaik
best_model = grid_search.best_estimator_

pipe = Pipeline(
    steps=[("preprocessor", scaler),
           ("model", best_model)]
)

# Menyimpan model terbaik dengan pickle
pklname = "generate_iris.pkl"

with open(pklname, 'wb') as file:
    pickle.dump(pipe, file)

# Melakukan pengujian
data = {'SepalLengthCm': 4,
        'SepalWidthCm': 3,
        'PetalLengthCm': 1,
        'PetalWidthCm': 3}
features = pd.DataFrame(data, index=[0])

with open(pklname, 'rb') as file:
    pick = pickle.load(file)

prediction = pick.predict(features)

result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
result[0]
if add_selectitem == "Iris species!":
    iris()
