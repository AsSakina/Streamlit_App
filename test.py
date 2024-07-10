#!pip install ydata_profiling
#!pip install streamlit
import pandas as pd
#from ydata_profiling import ProfileReport
df = pd.read_csv('Financial_inclusion_dataset.csv')
df.columns

from sklearn.preprocessing import LabelEncoder
## Encoder les caractéristiques catégorielles
label_encoders = {}
for column in df.columns[df.dtypes == 'object']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])
  
from sklearn.model_selection import train_test_split
# Séparer les features et la cible
X = df.drop(columns=['bank_account', 'uniqueid', 'country'])  # features
y = df['bank_account']  # cible

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Initialiser et entraîner le modèle
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = clf.predict(X_test)
accuracy_score= (y_test ,y_pred)
print(accuracy_score)

# Évaluation du modèle
print(classification_report(y_test, y_pred))

#!pip install streamlit
import streamlit as st

# Définir la fonction pour faire des prédictions
def predict_bank_account(year, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type):
    input_data = {'year': year,
                  'location_type': location_type,
                  'cellphone_access': cellphone_access,
                  'household_size': household_size,
                  'age_of_respondent': age_of_respondent,
                  'gender_of_respondent': gender_of_respondent,
                  'relationship_with_head': relationship_with_head,
                  'marital_status': marital_status,
                  'education_level': education_level,
                  'job_type': job_type}

    # Convertir les données en DataFrame
    input_df = pd.DataFrame([input_data])

    # Encoder les données catégorielles
    for column in input_df.columns[input_df.dtypes == 'object']:
        input_df[column] = label_encoders[column].transform(input_df[column])

    # Faire la prédiction
    prediction = clf.predict_proba(input_df)
    return prediction[0][1]  # Probabilité d'avoir un compte bancaire

# Titre de l'application
st.title('Prédiction d\'inclusion financière en Afrique de l\'Est')

# Ajouter des champs de saisie pour les caractéristiques
year = st.slider('Année', min_value=2016, max_value=2018)
location_type = st.selectbox('Type de localisation', ['Rural', 'Urban'])
cellphone_access = st.selectbox('Accès au téléphone portable', ['Yes', 'No'])
household_size = st.number_input('Taille du ménage', min_value=1)
age_of_respondent = st.number_input('Âge du répondant', min_value=16, max_value=100)
gender_of_respondent = st.selectbox('Genre du répondant', ['Male', 'Female'])
relationship_with_head = st.selectbox('Relation avec le chef de ménage', ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'])
marital_status = st.selectbox('Statut matrimonial', ['Married/Living together', 'Divorced/Seperated', 'Widowed', 'Single/Never Married'])
education_level = st.selectbox('Niveau d\'éducation', ['No formal education', 'Primary education', 'Secondary education', 'Vocational/Specialised training', 'Tertiary education', 'Other/Dont know/RTA'])
job_type = st.selectbox('Type d\'emploi', ['Formally employed Government', 'Formally employed Private', 'Informally employed', 'Self employed', 'Government Dependent', 'Other Income', 'No Income'])

# Bouton pour faire la prédiction
if st.button('Prédire'):
    prediction = predict_bank_account(year, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type)
    st.success(f'La probabilité d\'avoir un compte bancaire est : {prediction:.2%}')


