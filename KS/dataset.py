import pandas as pd
import numpy as np
import plotly as plt
import plotly.express as px
import streamlit as st
import matplotlib as plt
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import altair as alt
from streamlit_option_menu import option_menu 

data= pd.read_csv("malnutrition.csv")
data.head()
dta = pd.read_csv("sn.csv")
dta.head()
#data.info()
#data.describe()
data['city'] = data['regions'].rename('city')
data
# Example 5: When column names are different
dataset=pd.merge(data,dta, left_on='regions', right_on='city', how='left')
dataset
data.isnull().sum()

import pandas as pd
import plotly.express as px

# Exemple de données
#data = pd.DataFrame({
    #'latitude': [14.4974, 14.7100, 15.3833],
    #'longitude': [-14.4524, -17.4734, -16.4333],
    #'values': [10, 20, 30]  # Valeurs associées à chaque point
#})

#import nbformat
#nbformat.__version__

# Création de la carte avec Plotly
fig = px.scatter_mapbox(
    dataset, 
    lat=dataset["lat"], 
    lon=dataset["lng"], 
    size="Value",  # Taille des points selon la variable 'values'
    color="Value",  # Couleur des points selon la variable 'values'
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15, 
    zoom=6,  # Zoom sur le Sénégal
    mapbox_style="carto-positron"  # Style de la carte
)

# Titre de la carte
fig.update_layout(title_text="Carte du Sénégal avec variable 'values'", title_x=0.5 ) #height=nrows*500

fig.show()










#Page configuration
st.set_page_config(
    page_title="MALNUTRITION AU SENEGAL",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load the CSS file
local_css("style.css")


def add_custom_css():
    st.markdown(
        """
        <style>
        /* Change the background color of the sidebar */
        .sidebar .sidebar-content {
            background-color: #31333F;
        }
        /* Change the text color of the sidebar */
        .sidebar .sidebar-content {
            color: blue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# Call the functions to apply the customizations
add_custom_css()

header = st.container()
data = st.container()
with header : 
    #st.title()
    st.markdown("<h1 style='text-align: center; color: white;'> ""Malnutrition Au Senegal""</h1>", unsafe_allow_html=True) 
    st.toast("Welcome", icon="🫁")

#Titre de la page


#mettre une image sur le menu
st.sidebar.image("C:/Users/hp/Documents/malnutrition/m1.jpg")
#créer les options du menu
with st.sidebar:
    selected = option_menu("Main Menu", [ 'Graphic','Info','Correlation'], 
        icons=None, menu_icon=None, default_index=1, styles={"container": {"background-color": "#FAFAFA"},
        "nav-link-selected": {"background-color": "purple"}, 
        
        }
)



 #créer les choix  



#charger le dataframe et suprrimer les colonnes dont on a pas besoin
data= pd.read_csv('malnutrition.csv')
data.drop("Unit",axis=1,inplace=True)



#charger le dataframe sur streamlit
#st.dataframe(data)


#Choisir les options pour afficher les colonnes individuellement


#créer les graphic
fig1 = px.bar(data, x='Value', y='Date', title="Valeur de malnutrition par an")
fig2 = px.pie(data, values='Value', names='regions', title='Valeur de malnutrition dans les régions ')
fig3 = px.density_heatmap(data, x="Value", y="Date")
senegal_lat = 14.4974
senegal_long = -14.4524
fig4 = go.Figure(go.Scattermapbox(
    lat=[senegal_lat],  # Central point for map centering
    lon=[senegal_long ],
    mode='markers',
    marker=go.scattermapbox.Marker( size=5, color="red"),
    text=["regions"],
))
fig4.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=senegal_lat, lon=senegal_long),
        zoom=5.5,
    ),
    margin={"r":0,"t":0,"l":0,"b":0},
)

data['indicateurs'].head()

data["regions"].unique()

#choisir les options d'affichage dans le menu

if selected=="Graphic":
    df1=data.groupby(['regions','Date']).sum()
    exp=st.expander("malnutriton of child in Senegal")
    with exp:
        col1,col2= st.columns(2)    
        choix= col2.selectbox("Sélectionnez une régions",['Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kédougou',
                'Kolda', 'Louga', 'Matam', 'Saint-Louis', 'Sédhiou', 'Tambacounda',
                'Thiès', 'Ziguinchor', 'SENEGAL'])
        date=col1.radio('Sélectionnez une année',options=['2018','2019'],horizontal=True)

        df2= df1.loc[choix, :]
        df3= df2.filter(like= date, axis= 0).values
        st.markdown(f"""
                        <div style="border: 3px solid green; padding: 10px; border-radius: 5px;text-align: center;" ,width=500>
                                <write style ="text-align: center" >
                                Valeur de  taux du malnutrition à {choix }<br/>
                                🍼 {df3[0,1]}
                                </write>
                            </div>
                            """, unsafe_allow_html=True)
    st.markdown ("<h2 style='text-align: left; color: white;'> ""Graphic""</h2>", unsafe_allow_html=True)

        
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig4)
elif selected=="Info":
    st.markdown("<h1 style='text-align: left; color: white;'> ""Informations""</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left; color: white;'> ""La Malnutrition aigue ""</h2>", unsafe_allow_html=True)
    st.write(
    "Au niveau national, La malnutrition aigue est de 9% mais on note une "
    "disparité au niveau des régions du pays. Les régions du "
    "nord du Sénégal semblent être les plus affectées : les "
    "enfants des régions de Matam (16,5%), de Louga (16,1%) et "
    "de Saint-Louis (14,7%) ainsi que ceux du département de "
    "Podor (18,2%) sont en situation préoccupante. De plus, les "
    "prévalences de MAS dépassent le seuil d’urgence de 2% "
    "dans ces 3 régions ainsi que dans la région de "
    "Tambacounda (2,1%). (SMART 2015)."
    
)
    

    st.markdown ("<h2 style='text-align: left; color: white;'> ""La Malnutrition chronique""</h2>", unsafe_allow_html=True)
    st.write("La malnutrition chronique est un problème de santé, "
    "Il résulte d’une alimentation inadéquate pendant une longue "
    "durée. Au Sénégal, la prévalence de cette forme de "
    "malnutrition indique une situation nutritionnelle acceptable, "
    "avec un taux 17,1% chez les enfants de 0-59 mois."
)
    st.markdown ("<h2 style='text-align: left; color: white;'> ""L’insuffisance pondérale""</h2>", unsafe_allow_html=True)
    st.write(
    "C’est un indicateur qui reflète aussi bien une malnutrition "
    "chronique qu’une malnutrition aigüe. Au Sénégal, la "
    "prévalence de 13,9% de l’insuffisance pondérale retrouvée "
    "au niveau national révèle une situation précaire chez les "
    "enfants de 0-59 mois. Elle est légèrement plus élevée chez "
    "les garçons (14 %) que chez les filles (12 %)."

)
    st.markdown ("<h2 style='text-align: left; color: white;'> ""Moyen de Prévention contre la malnutrition""</h2>", unsafe_allow_html=True)
    st.write("1. Éducation nutritionnelle : Sensibiliser les communautés à l'importance d'une alimentation équilibrée et diversifiée, incluant des protéines, des légumes, des fruits et des céréales complètes.")
    st.write("2. Allaitement maternel exclusif : Encourager l’allaitement maternel exclusif pendant les six premiers mois de vie, car il fournit tous les nutriments nécessaires et protège contre les infections.")
    st.write("3. Accès à des aliments variés et nutritifs : Garantir que les familles aient accès à une alimentation suffisante et équilibrée, avec des produits locaux, frais et riches en nutriments essentiels.")
    st.write("4. Lutte contre la pauvreté et l’insécurité alimentaire : Réduire la pauvreté et améliorer les revenus des familles pour leur permettre d'acheter ou de produire suffisamment de nourriture.")
    st.write("5. Accès aux soins de santé et surveillance de la croissance : Offrir un suivi médical régulier pour détecter et traiter les signes de malnutrition, en particulier chez les enfants et les femmes enceintes.")

    st.image('C:/Users/hp/Documents/malnutrition/info.jpg')
elif selected=="Correlation":
    st.markdown ("<h2 style='text-align: left; color: white;'> ""Correlation entre les tables""</h2>", unsafe_allow_html=True)
    st.plotly_chart(fig3)