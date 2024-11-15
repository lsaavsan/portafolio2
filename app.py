# Importamos las bibliotecas
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
from scipy.stats import gaussian_kde

# Configuración de la página
st.set_page_config(
    page_title="Mi Aplicación",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

df = pd.read_excel('titanic.xlsx')

# Definimos el título y la descripción de la aplicación
st.markdown(
    """
        <h1 style="font-size:20pt;color:darkblue;font-weight:bold;text-align:center;margin:0px;border:0px;"> Descifrando el Titanic: <br> Supervivencia y Predicción con Machine Learning</h1>
    """, 
    unsafe_allow_html=True
)
# Cargamos y mostramos el logo
# Ruta de la imagen


# Función para convertir la imagen a base64
def get_image_base64(img_path):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

logo_path = "./titanic.png"
# Convertir la imagen a base64
img_base64 = get_image_base64(logo_path)

# Mostrar la imagen centrada
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{img_base64}" width="500">
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("""
            A las 23:40 del 14 de abril de 1912, el Titanic chocó contra un Iceberg. La colisión abrió varias planchas del 
            casco en su lado de estribor bajo la línea de flotación. Durante dos horas y media el barco se fue hundiendo 
            gradualmente por su parte delantera mientras la popa se elevaba, y en este tiempo varios cientos de pasajeros 
            y tripulantes fueron evacuados en los botes salvavidas.
            """)
st.markdown(
    """
            <h1 style="font-size:16pt;color:darkgreen;font-weight:bold;">Variables incluidas en el Dataset</h1>
            <ul style="font-size:10pt;text-align:justify;line-height:1.5;">
            <li><strong>PassengerId:</strong> Id del pasajero.</li>
            <li><strong>Name:</strong> Nombre del pasajero, almacenado en String. </li>
            <li><strong>Pclass:</strong> Para indicar la clase en la que viajaba la persona: Clase 1, Clase 2 y Clase 3. </li>
            <li><strong>Sex:</strong> Sexo del pasajero, male o famale. </li>
            <li><strong>Age:</strong> Edad del pasajero, almacenado con enteros. </li>
            <li><strong>Sibsp:</strong> Número de hermanos, que el pasajero, que estaban a bordo. Almacenado en un entero. </li>
            <li><strong>Parch:</strong> Número de padres, del pasajero, que estaban a bordo. Almacenados en un entero.</li> 
            <li><strong>Embarked:</strong> Indica el puerto de embarque: cherbourg, Queenstown y Southampton.</li>
            <li><strong>Fare:</strong> Indica el monto, en libras esterlinas, que el pasajero pago para obtener su boleto. Almacenado en un double. </li>   
            <li><strong>Ticket:</strong> Número de ticket que el pasajero entregó al abordar.</li> 
            <li><strong>Cabin:</strong> Indica la cabina que fue asignada al pasajero, almacenada en un String.</li> 
            <li><strong>Survived:</strong> Indica si la persona sobrevivió o no al naufragio.</li>
            </ul>""", 
    unsafe_allow_html=True
)
st.markdown(
    """
    <hr style="border: 3px solid black; margin: 20px 0;">
    """, 
    unsafe_allow_html=True
)

st.sidebar.header('Características a clasificar')

# Cargar modelo preentrenado, codificador, escalador y DataFrame de características de entrenamiento
load_clf = pickle.load(open('rftitanic.pkl', 'rb'))
encoder = pickle.load(open('encoder_titanic.pkl', 'rb'))
scaler = pickle.load(open('scaler_titanic.pkl', 'rb'))
X_train_prepared = pickle.load(open('X_train_prepared.pkl', 'rb'))  # DataFrame de referencia
encoder1 = pickle.load(open('encoder1_titanic.pkl', 'rb'))

# Función para obtener las características del usuario
def user_input_features():
    Pclass = st.sidebar.selectbox('Clase del pasajero', ('Clase 1', 'Clase 2', 'Clase 3'))
    Sex = st.sidebar.selectbox('Sexo del pasajero', ('Masculino', 'Femenino'))
    Age = st.sidebar.slider('Edad del pasajero', 0, 80, 40)
    SibSp = st.sidebar.slider('Número de Hermanos del pasajero', 0, 8, 1) 
    Parch = st.sidebar.slider('Número de Padres de pasajero', 0, 6, 1)
    Embarked = st.sidebar.selectbox('Puerto de embarque del pasajero', ('Southampton', 'Cherbourg', 'Queenstown')) 
    Fare = st.sidebar.slider('Tarifa pagada por el pasajero', 0, 512, 7) 


    # Creamos el diccionario
    data = {
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Embarked': [Embarked],
        'Fare': [Fare]
    }
    # Convertimos el diccionario en un DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Cargamos el archivo o pedimos datos del usuario
input_df = user_input_features()

# Codificación OneHot con el codificador cargado
categorical_features = input_df[['Pclass', 'Sex', 'Embarked']]
#categorical_features = categorical_features[['Pclass', 'Sex', 'Embarked']]
encoded_features = encoder.transform(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Pclass', 'Sex', 'Embarked']))

# Escalar características numéricas con el escalador cargado
numerical_features = input_df[['Age', 'SibSp', 'Parch', 'Fare']]
scaled_numerical = scaler.transform(numerical_features)
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features.columns)

# Concatenar datos procesados
input_df_processed = pd.concat([scaled_numerical_df, encoded_df], axis=1)

# Alinear y rellenar `input_df_processed` para que coincida con `X_train_prepared`
input_df_processed = input_df_processed.reindex(columns=X_train_prepared.columns, fill_value=0)

# Mostrar los datos ingresados
st.markdown(
    """
        <h1 style="font-size:18pt;color:red;font-weight:bold;text-align:left;margin:0px;border:0px;"> Características a clasificar</h1>
    """, 
    unsafe_allow_html=True
)
st.write(input_df)


# Realizar la predicción con el modelo cargado
prediction = load_clf.predict(input_df_processed)
prediction_proba = load_clf.predict_proba(input_df_processed)
etiquetas = encoder1.inverse_transform(np.arange(len(encoder1.classes_)))
proba_df = pd.DataFrame(prediction_proba, columns=etiquetas)

# Mostrar los resultados
col1, col2 = st.columns(2)
with col1:
    st.markdown(
    """
        <h1 style="font-size:18pt;color:red;font-weight:bold;text-align:left;margin:0px;border:0px;"> Predicción</h1>
    """, 
    unsafe_allow_html=True)
    st.write('Sobrevivió' if prediction[0] == 1 else 'No sobrevivió')

with col2:
    st.markdown(
    """
        <h1 style="font-size:18pt;color:red;font-weight:bold;text-align:left;margin:0px;border:0px;"> Probabilidad de predicción</h1>
    """, 
    unsafe_allow_html=True
)
    st.write(proba_df)


st.markdown(
    """
    <hr style="border: 3px solid black; margin: 20px 0;">
    """, 
    unsafe_allow_html=True
)





st.markdown("""

<p  style="font-size:20pt;color:darkblue;line-height:1.5;font-weight:bold;text-align:center;">Análisis Exploratorio de Datos (EDA)</p>
""", 
    unsafe_allow_html=True
)


# Ordenar el DataFrame y restaurar el índice
df_tarifa_sorted = df.sort_values(by='Fare').reset_index(drop=True)

# Crear el gráfico de líneas usando Plotly
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_tarifa_sorted.index,         # Índice de pasajero en el eje X
        y=df_tarifa_sorted['Fare'],       # Tarifa en el eje Y
        mode='lines',                     # Tipo de gráfico de líneas
        line=dict(color='orangered', width=2), # Color y grosor de la línea
        opacity=0.8                       # Transparencia de la línea
    )
)

# Configuración del título y etiquetas de los ejes
fig.update_layout(
    title='Tarifas pagadas por los Pasajeros del Titanic',
    xaxis_title='Índice de Pasajero',
    yaxis_title='Tarifa',
    width=1000,
    height=700
)

# Configurar los valores del eje Y (tarifas) en incrementos de 20
fig.update_yaxes(tick0=0, dtick=20, range=[0, df_tarifa_sorted['Fare'].max() + 20])

# Configurar los valores del eje X (índice de pasajero) en incrementos de 50
fig.update_xaxes(tick0=0, dtick=50)

# Mostrar la cuadrícula para mejorar la legibilidad del gráfico
fig.update_layout(
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# Mostrar el gráfico
st.plotly_chart(fig)


st.markdown("""
                <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación las tarifas pagadas por los pasajeros</h1>
                <ul style="font-size:12pt;text-align:justify;line-height:1.5;">

                <li><strong>Total de pasajeros:</strong> Son 891 pasajeros del Titanic.</li>
                <li><strong>Tarifas bajas predominantes:</strong> La mayoría de las tarifas pagadas por los pasajeros del Titanic se encuentran en el rango bajo,  entre 0 y 20 libras esterlinas aproximadamente. Esto es visible en el tramo inicial del gráfico, donde la línea se mantiene casi horizontal entre el índice de pasajero 0 hasta el índice 520 aproximadamente.</li>

                <li><strong>Aumento gradual en las tarifas:</strong> A partir del índice de pasajero 520, las tarifas comienzan a aumentar hasta 40 libras esterlinas aproximadamente hasta el índice 700. Después de esto las tarifas comienzan a subir de manera pronunciada hasta el índice de pasajeros 850 llegando a pagar hasta 120 libras esterlinas.</li>

                <li><strong>Valores muy altos:</strong> A partir del índice de pasajeros 850 la línea aumenta abruptamente alcanzando tarifas de hasta 512 libras esterlinas.</li>
                </ul>
                """,
                unsafe_allow_html=True)

st.markdown(
    """<hr style="border: 3px solid black; margin: 20px 0;">""", 
    unsafe_allow_html=True
)

# Cuenta la cantidad de pasajeros que embarcaron en cada puerto y almacena el resultado en 'df_puerto'
df_puerto = df['Embarked'].value_counts()

# Crear el gráfico de pastel usando Plotly
fig = px.pie(
    df_puerto,
    values=df_puerto.values,
    names=df_puerto.index,
    title="Distribución de Pasajeros por Puerto de Embarque",
    hole=0,  # Para pastel completo, sin agujero en el centro
)

# Configurar el formato de porcentaje y agregar sombra en la gráfica
fig.update_traces(
    textinfo='percent+label',    # Muestra el porcentaje y la etiqueta
    hoverinfo='label+percent',   # Muestra la etiqueta y el porcentaje al pasar el cursor
    pull=[0.05] * len(df_puerto), # Opcional: resalta ligeramente cada porción
)

# Agrega la leyenda
fig.update_layout(
    legend_title="Puerto de Embarque",
    legend=dict(x=1, y=0.9)  # Ajusta la posición de la leyenda
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre los puertos en los que se embarcaron los pasajeros</h1>
            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
            <li><strong>Concentración en Southampton:</strong> La gran mayoría de los pasajeros embarcaron en Southampton (72.44%), lo que sugiere que la demanda de este viaje era más alta en el Reino Unido, de donde el Titanic inició su viaje.</li>
            <li><strong>Distribución Menor en Otros Puertos:</strong> Cherbourg(18.90%) y Queenstown(8.66%) contribuyeron con una menor cantidad de pasajeros en comparación con Southampton.</li>
            </ul>
    """, 
    unsafe_allow_html=True
)


st.markdown(
    """<hr style="border: 3px solid black; margin: 20px 0;">""", 
    unsafe_allow_html=True
)

# Cuenta la cantidad de pasajeros de cada sexo
df_sexo = df['Sex'].value_counts()

# Crear el gráfico de barras usando Plotly
fig = px.bar(
    df_sexo,
    x=df_sexo.index,                   # El sexo como eje X
    y=df_sexo.values,                  # La frecuencia como eje Y
    labels={'x': 'Sexo', 'y': 'Frecuencia de Pasajeros'},
    title='Pasajeros del Titanic por Sexo',
    text=df_sexo.values                # Muestra el valor numérico encima de las barras
)

# Personalizar los colores de las barras
fig.update_traces(
    marker_color=['green', 'red'],  # Verde para un grupo y rojo para otro
    opacity=0.5,                   # Transparencia de las barras
    textposition='outside'         # Posición del texto fuera de las barras
)

# Ajustar la cuadrícula y los ejes
fig.update_layout(
    yaxis=dict(
        showgrid=True,              # Activa la cuadrícula del eje Y
        tick0=0,
        dtick=50,                   # Ticks del eje Y cada 50 unidades
        title="Frecuencia de Pasajeros"
    ),
    xaxis=dict(
        showgrid=True,              # Activa la cuadrícula del eje X
        title="Sexo"
    ),
    showlegend=False               # Oculta la leyenda porque los valores son evidentes
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre el sexo de los pasajeros</h1>
            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
            <li><strong>Predominancia de pasajeros hombres:</strong> La barra verde representa a los pasajeros hombres y muestra que habían  700 hombres a bordo, es decir la mayoría de los pasajeros en el Titanic eran hombres.
            <li><strong>Menor cantidad de pasajeros mujeres:</strong> La barra roja representa a las pasajeros mujeres, con una cantidad 314, es decir aproximadamente el doble de hombres que de mujeres. Esta diferencia en la distribución por sexo puede ser relevante para analizar otros aspectos, como las tasas de supervivencia, ya que es conocido que se dio prioridad a mujeres y niños en los botes salvavidas.
            </ul>
            """, 
            unsafe_allow_html=True
)
st.markdown(
    """<hr style="border: 3px solid black; margin: 20px 0;">""", 
    unsafe_allow_html=True
)

# Cuenta la cantidad de pasajeros según su estado de supervivencia
df_sobrevivencia = df['Survived'].value_counts()

# Crear el gráfico de barras horizontal usando Plotly
fig = px.bar(
    df_sobrevivencia,
    y=df_sobrevivencia.index,            # El estado de supervivencia como eje Y
    x=df_sobrevivencia.values,           # La frecuencia como eje X
    labels={'y': 'Sobrevivencia', 'x': 'Frecuencia de Pasajeros'},
    title='Distribución de Pasajeros por Estado de Supervivencia',
    text=df_sobrevivencia.values         # Muestra el valor numérico dentro de las barras
)

# Personalizar los colores de las barras
fig.update_traces(
    marker_color=['blue', 'orange'],  # Azul para un grupo, naranja para otro
    opacity=0.5,                     # Transparencia de las barras
    textposition='outside'           # Posición del texto fuera de las barras
)

# Ajustar la cuadrícula y los ejes
fig.update_layout(
    xaxis=dict(
        showgrid=True,               # Activa la cuadrícula del eje X
        tick0=0,
        dtick=50,                    # Ticks del eje X cada 50 unidades
        title="Frecuencia de Pasajeros"
    ),
    yaxis=dict(
        showgrid=False,              # Oculta la cuadrícula del eje Y
        title="Sobrevivencia",
        tickvals=df_sobrevivencia.index,
        ticktext=["No Sobrevivió", "Sobrevivió"]  # Etiquetas personalizadas para 0 y 1
    ),
    showlegend=False                # Oculta la leyenda porque los valores son evidentes
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)



st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre sobrevivencia de los pasajeros</h1>
            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
            <li><strong>Pasajeros que murieron:</strong> La barra azul representa la cantidad de pasajeros que murieron al hundimiento del Titanic.
            Esta barra se extiende hasta 549, indicando que la mayoría de los pasajeros en el Titanic no sobrevivieron.</li>
            <li><strong>Pasajeros que sobrevivieron ("Sobrevivió"):</strong> La barra amarilla representa la cantidad de pasajeros que lograron sobrevivir.
            Esta barra es más corta que la azul, alcanzando 342, lo que sugiere que una menor cantidad de pasajeros logró sobrevivir en comparación con los que no lo hicieron.</li>
            </ul>
    """, 
    unsafe_allow_html=True
)


st.markdown(
    """<hr style="border: 3px solid black; margin: 20px 0;">""", 
    unsafe_allow_html=True
)

# Crear un gráfico de caja (boxplot) usando Plotly
fig = px.box(
    df,
    x="Survived",             # Eje X representa el estado de supervivencia (0 = Murió, 1 = Sobrevivió)
    y="Age",                  # Eje Y representa la edad de los pasajeros
    color="Sex",              # Diferenciar los datos por el sexo (Masculino o Femenino)
    color_discrete_sequence=px.colors.qualitative.Set1,  # Paleta de colores similar a la de Seaborn "Set1"
    labels={"Survived": "Sobrevivencia", "Age": "Edad"}, # Etiquetas de los ejes
    title="Distribución de Edad según Supervivencia y Sexo"
)

# Configurar el rango del eje Y con incrementos de 5
fig.update_yaxes(
    tick0=0,
    dtick=5,
    range=[0, df["Age"].max() + 5]
)

# Configurar la leyenda
fig.update_layout(
    legend_title_text="Sexo",
    legend=dict(x=0.89, y=1.0),  # Ajuste de posición similar a `bbox_to_anchor`
    xaxis=dict(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["No Sobrevivió", "Sobrevivió"]
    )
)

# Activar la cuadrícula en el gráfico
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)












st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre sobrevivencia de los pasajeros por edad y sexo</h1>
            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
            <li>Este gráfico de caja muestra la distribución de la edad de los pasajeros del Titanic según sobrevivencia y está diferenciado por sexo.</li>
            <li><strong>Cajas (box) y Bigotes (whiskers):</strong> Las cajas representan el rango intercuartílico, que incluye al 50% de las edades centrales en cada grupo. La línea dentro de cada caja es la mediana. Los bigotes se extienden hasta los valores mínimo y máximo dentro de 1.5 veces el IQR, y los puntos fuera de este rango son valores atípicos (outliers).</li>
            <li><strong>Hombres que murieron:</strong>  La mediana de edad es 29 años, el 50% de las edades centrales estuvo entre 21 y 39 años.</li>
            <li><strong>Mujeres que murieron:</strong>  La mediana de edad es 24 años, el 50% de las edades centrales estuvo entre 16 y 33 años.</li>
            <li><strong>Hombres que sobrevivieron:</strong>  La mediana de edad es 28 años, el 50% de las edades centrales estuvo entre 18 y 36 años.</li>
            Mujeres que sobrevivieron:</strong>  La mediana de edad es 28 años, el 50% de las edades centrales estuvo entre 19 y 38 años.</li>
            <li><strong>Valores Atípicos:</strong>  Se observan valores atípicos en la parte superior del gráfico en hombres con edades mayores de 65 años, principalmente en los murieron.</li>
            </ul>
    """, 
    unsafe_allow_html=True
)



st.markdown(
    """<hr style="border: 3px solid black; margin: 20px 0;">""", 
    unsafe_allow_html=True
)

import plotly.express as px
import streamlit as st

# Definir colores personalizados para "Masculino" y "Femenino"
color_map = {
    "Clase 1": "darkblue",  # Color para "Masculino"
    "Clase 2": "darkgreen",       # Color para "Femenino"
    "Clase 3": "red",       # Color para "Femenino"

}

# Crear gráfico de violín utilizando Plotly Express
fig = px.violin(
    df,
    x="Survived",               # Eje X representa el estado de supervivencia
    y="Fare",                    # Eje Y representa la edad de los pasajeros
    color="Pclass",                # Diferenciar por sexo
    box=True,                   # Añade un gráfico de caja dentro del violín
    points=False,               # No muestra puntos individuales
    color_discrete_map=color_map,  # Mapa de colores personalizado para "Masculino" y "Femenino"
    labels={"Survived": "Sobrevivencia", "Fare": "Tarifa"},  # Etiquetas personalizadas
    title="Distribución de la Edad según Supervivencia y Clase",
    facet_col="Pclass",             # Divide los violines en columnas separadas por sexo
    facet_col_spacing=0.1         # Espaciado entre columnas (reduce o aumenta según necesites)
)

# Configurar los ticks y rango del eje Y
fig.update_yaxes(
    tick0=0,
    dtick=50,                      # Incrementos de 5 en el eje Y
    range=[-50, df["Fare"].max() + 50]
)

# Configurar los ticks y etiquetas del eje X
fig.update_xaxes(
    tickmode="array",
    tickvals=[0, 1],
    ticktext=["No Sobrevivió", "Sobrevivió"]
)

# Ajustar el ancho del violín y evitar solapamiento de la caja
fig.update_traces(
    width=0.2,                  # Ajusta el ancho de cada violín
    box_width=0.1,              # Ajusta el ancho de las cajas para que encajen dentro del violín
    meanline_visible=True       # Opcional: Mostrar la línea de la media
)

# Configurar el layout
fig.update_layout(
    height=800,                  # Altura del gráfico
    width=1500,                  # Ancho del gráfico
    legend_title="Clase",          # Título de la leyenda
    xaxis_title="Sobrevivencia",  # Etiqueta del eje X
    yaxis_title="Tarifa",            # Etiqueta del eje Y


)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


st.markdown(
    """
                <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre sobrevivencia de los pasajeros por tarifa y clase</h1>
                <ul style="font-size:12pt;text-align:justify;line-height:1.5;">

                <li>Cada caja representa la distribución de las tarifas pagadas dentro de cada clase y según si el pasajero sobrevivió o no.</li>
                <li><strong>Los pasajeros de la Clase 1 que sobrevivieron</strong> pagaron tarifas más altas en general, 
                la mediana de tarifa es 77 libras esterlinas, el 50% de las tarifas centrales estuvo entre las 50 y 112 libras esterlinas aproximadamente, 
                pero se pagaron tarifas que superan las 500 libras esterlinas. Mientras que <strong> los pasajeros de la Clase 1 que murieron</strong> 
                también tienden a haber pagado tarifas elevadas, la mediana de tarifa es 44 libras esterlinas, el 50% de las tarifas centrales estuvo 
                entre las 27 y 79 libras esterlinas aproximadamente, pero se pagaron tarifas que superan las 250 libras esterlinas.</li>

                <li><strong>Los pasajeros de la Clase 2 que sobrevivieron</strong> pagaron tarifas bajas, la mediana de tarifa es 21 libras esterlinas, 
                el 50% de las tarifas centrales estuvo entre las 13 y 26 libras esterlinas aproximadamente, pero se pagaron tarifas de 50 libras esterlinas 
                aproximadamente. Mientras que <strong> los pasajeros de la Clase 2 que murieron</strong> también tienden a haber pagado tarifas bajas, 
                la mediana de tarifa es 13 libras esterlinas, el 50% de las tarifas centrales estuvo entre las 11 y 26 libras esterlinas aproximadamente, 
                pero se pagaron tarifas que llegan a las 75 libras esterlinas aproximadamente.</li>

                <li><strong>Los pasajeros de la Clase 3 que sobrevivieron</strong> pagaron las tarifas más bajas, la mediana de tarifa es 8 libras esterlinas, 
                el 50% de las tarifas centrales estuvo entre las 7 y 15 libras esterlinas aproximadamente, pero se pagaron tarifas de 50 libras esterlinas 
                aproximadamente. Mientras que <strong> los pasajeros de la Clase 3 que murieron</strong> también pagaron tarifas más bajas, la mediana de 
                tarifa es 8 libras esterlinas, el 50% de las tarifas centrales estuvo entre las 7 y 15 libras esterlinas aproximadamente, pero se pagaron 
                tarifas cercanas a la 70 libras esterlinas aproximadamente.</li>
                <li><strong>Clase 1</strong> muestra una clara ventaja en términos de supervivencia, especialmente para aquellos que pagaron tarifas más altas. Sin embargo, no todos los que pagaron tarifas elevadas lograron sobrevivir, como lo indica la presencia de una caja amplia tanto en los que murieron como en los que sobrevivieron.</li>
                <li><strong>Clase 2 y Clase 3</strong> presentan patrones similares, donde la mayoría de las tarifas son bajas. Los pasajeros de Clase 3 parecen haber tenido la peor probabilidad de supervivencia.</li>
                </ul>
    """, 
    unsafe_allow_html=True
)



st.markdown(
    """<hr style="border: 3px solid black; margin: 20px 0;">""", 
    unsafe_allow_html=True
)


# Crear un gráfico de dispersión utilizando Plotly Express
fig = px.scatter(
    df,
    x="Age",                     # Eje X representa la edad de los pasajeros
    y="Fare",                    # Eje Y representa la tarifa pagada por los pasajeros
    color="Survived",            # Diferenciar los puntos por el estado de supervivencia
    color_discrete_sequence=px.colors.qualitative.Set2,  # Paleta de colores similar a Seaborn "Set2"
    labels={"Age": "Edad", "Fare": "Tarifa", "Survived": "Sobrevivencia"},  # Etiquetas personalizadas
    title="Relación entre Edad y Tarifa pagada según Supervivencia",
    hover_data=["Pclass", "Sex"]  # Información adicional al pasar el cursor
)

# Configurar los valores del eje Y (tarifa) en incrementos de 25
fig.update_yaxes(
    tick0=0,
    dtick=25,                     # Incrementos de 25 en el eje Y
    title="Tarifa",               # Etiqueta del eje Y
    showgrid=True                 # Activar cuadrícula para el eje Y
)

# Configurar los valores del eje X (edad) en incrementos de 5
fig.update_xaxes(
    tick0=0,
    dtick=5,                      # Incrementos de 5 en el eje X
    title="Edad",                 # Etiqueta del eje X
    showgrid=True                 # Activar cuadrícula para el eje X
)

# Ajustar el diseño del gráfico
fig.update_layout(
    legend_title="Sobrevivencia",  # Título de la leyenda
    height=600,                    # Altura del gráfico
    width=800                      # Ancho del gráfico
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre sobrevivencia de los pasajeros por tarifa y Edad</h1>
            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
            <li><strong>Edad y Supervivencia:</strong> Los niños hasta 5 años parecen tener una mayor probabilidad de supervivencia que los adultos y ancianos, lo cual puede reflejar el esfuerzo por salvar a los más vulnerables.</li>
            <li><strong>Tarifa y Supervivencia:</strong> Las tarifas más altas parecen estar asociadas con una mayor probabilidad de supervivencia. Esto probablemente se debe a la ubicación de los camarotes y el acceso preferente a los recursos de salvamento.</li>
            <li><strong>Tarifas bajas y mortalidad:</strong> Los pasajeros que pagaron tarifas más bajas parecen haber tenido menos oportunidades de sobrevivir.</li>
            </ul>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <hr style="border: 3px solid black; margin: 20px 0;">
    """, 
    unsafe_allow_html=True
)



# Calcular la densidad KDE
age_data = df["Age"].dropna()  # Remover valores nulos
kde = gaussian_kde(age_data)   # Estimación de densidad
x_vals = np.linspace(age_data.min(), age_data.max(), 500)  # Valores de x para la línea KDE
y_vals = kde(x_vals)           # Densidad para cada valor de x

# Crear el histograma
histogram = go.Histogram(
    x=age_data,
    nbinsx=50,               # Número de bins
    marker_color="green",      # Color de las barras
    opacity=0.8,             # Transparencia de las barras
    name="Histograma"        # Nombre para la leyenda
)

# Crear la línea de densidad
kde_line = go.Scatter(
    x=x_vals,
    y=y_vals * len(age_data) * (age_data.max() - age_data.min()) / 50,  # Escalar la densidad
    mode="lines",
    line=dict(color="darkgreen", width=2),  # Estilo de la línea
    name="KDE (Densidad)"             # Nombre para la leyenda
)

# Combinar ambos trazos
fig = go.Figure(data=[histogram, kde_line])

# Configurar los ejes
fig.update_xaxes(
    title="Edad",
    tick0=0,
    dtick=5,
    showgrid=True
)
fig.update_yaxes(
    title="Frecuencia",
    tick0=0,
    dtick=5,
    showgrid=True
)

# Configurar el layout
fig.update_layout(
    title="Distribución de Edad de los Pasajeros del Titanic",
    height=600,
    width=800,
    bargap=0.1,
    legend_title="Elementos"
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación sobre sobrevivencia de los pasajeros por tarifa y Edad</h1>
            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
            <li><strong>Pasajeros jóvenes y adultos jóvenes predominaban:</strong> La mayor parte de los pasajeros se concentraba en el rango de 15 a 40 años, siendo el grupo de 20 a 30 años el más común.</li>
            <li><strong>Menos pasajeros mayores:</strong> A medida que la edad avanza, la cantidad de pasajeros disminuye, lo cual se puede notar en la curva descendente y la disminución de las barras a partir de los 40 años.</li>
            <li><strong>Niños presentes pero en menor número:</strong> Hay un número notable de pasajeros menores de 10 años, lo cual muestra que había un número significativo de familias a bordo.</li>
            </ul>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <hr style="border: 3px solid black; margin: 20px 0;">
    """, 
    unsafe_allow_html=True
)


# Crear gráfico de barras agrupadas y divididas por clase
fig = px.histogram(
    df,
    x="Survived",                   # Eje X: Supervivencia
    color="Sex",                    # Diferenciar por sexo
    facet_col="Pclass",             # Dividir por clase en columnas
    color_discrete_map={"Masculino": "blue", "Femenino": "green"},  # Colores personalizados
    barmode="group",                # Barras agrupadas
    text_auto=True                  # Mostrar valores en las barras
)

# Configurar los ejes
fig.update_xaxes(
    title="Sobrevivencia",          # Etiqueta del eje X
    tickvals=[0, 1],                # Valores en el eje X
    ticktext=["No Sobrevivió", "Sobrevivió"]  # Etiquetas de los valores
)
fig.update_yaxes(
    title="Cantidad",               # Etiqueta del eje Y
    showgrid=True                   # Mostrar cuadrícula
)

# Configurar el diseño
fig.update_layout(
    title="Distribución de Supervivencia por Sexo y Clase",
    height=600,                     # Altura del gráfico
    width=1000,                     # Ancho del gráfico
    legend_title="Sexo",            # Título de la leyenda
    bargap=0.1                      # Espaciado entre las barras
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)












st.markdown(
    """
            <h1 style="font-size:14pt;color:red;font-weight:bold;">Interpretación de la Distribución de Supervivencia por Clase y Sexo</h1>
            <p style="font-size:12pt;text-align:justify;line-height:1.5;">
                El gráfico muestra la distribución de pasajeros que sobrevivieron y murieron en el Titanic, separados por <strong>clase de pasajero</strong> (<em>Pclass</em>) y <strong>sexo</strong> (<em>Sex</em>). Cada columna representa una clase distinta, y dentro de cada clase, los pasajeros se agrupan según su estado de supervivencia (murió o sobrevivió) y se diferencian por el color de las barras para reflejar el sexo.
            </p>

            <ul style="font-size:12pt;text-align:justify;line-height:1.5;">
                <li><strong>Clase 3:</strong> La mayoría de los pasajeros en esta clase no sobrevivieron, especialmente los hombres, que tienen una cantidad significativamente mayor de fallecidos (en azul) en comparación con las mujeres (en verde). Sin embargo, también hay una pequeña cantidad de supervivientes en ambas categorías de sexo.</li>
                <li><strong>Clase 1:</strong> En esta clase, la distribución es más equilibrada. Las mujeres tienen una mayor tasa de supervivencia que los hombres. La cantidad de mujeres supervivientes es notablemente mayor que la de hombres en esta clase.</li>
                <li><strong>Clase 2:</strong> En esta clase, al igual que en la Clase 1, las mujeres tienen una mayor tasa de supervivencia que los hombres. Aunque el número de fallecidos es menor en esta clase, se observa una proporción similar donde las mujeres tienen una mayor posibilidad de supervivencia.</li>
            </ul>

            <p style="font-size:12pt;text-align:justify;line-height:1.5;">
                <strong>Conclusión:</strong> Los datos sugieren que las mujeres en las clases 1 y 2 tenían mayores probabilidades de supervivencia en comparación con los hombres. En la clase 3, una gran cantidad de hombres fallecieron, lo que puede indicar que los pasajeros de clase baja tuvieron menos probabilidades de supervivencia en general. Esto refleja las prácticas y prioridades de evacuación del Titanic, donde se dio preferencia a mujeres y niños, especialmente en las clases superiores.
            </p>

    """, 
    unsafe_allow_html=True
)


