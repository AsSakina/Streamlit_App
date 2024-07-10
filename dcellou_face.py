!pip install streamlit
import streamlit as st
import cv2
#%%writefile app.py

# Charger le modèle cascade pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fonction pour détecter les visages
def detect_faces(image, scaleFactor, minNeighbors):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return faces

# Fonction pour dessiner et enregistrer l'image avec les visages détectés
def save_image_with_faces(image, faces, output_path):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Dessiner un rectangle bleu autour du visage
    cv2.imwrite(output_path, image)

# Interface utilisateur avec Streamlit
def main():
    st.title('Détection de visages avec OpenCV et Streamlit')

    uploaded_file = st.file_uploader("Uploader une image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption='Image originale')

        # Paramètres ajustables
        scaleFactor = st.slider('scaleFactor', 1.1, 2.0, 1.2, step=0.1)
        minNeighbors = st.slider('minNeighbors', 1, 10, 3)
        color = st.color_picker('Choisir la couleur du rectangle', '#ff0000')  # Par défaut, rouge

        if st.button('Détecter les visages'):
            faces = detect_faces(image, scaleFactor, minNeighbors)
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            st.image(image, channels="BGR", caption='Image avec visages détectés')

            if st.button('Télécharger image avec visages détectés'):
                output_path = 'image_with_faces.jpg'  # Nom du fichier de sortie
                save_image_with_faces(image, faces, output_path)
                st.success(f'L\'image avec visages détectés a été enregistrée sous {output_path}')

if __name__ == '__main__':
    main()
