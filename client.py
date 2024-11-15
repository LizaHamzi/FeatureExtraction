import streamlit as st
import numpy as np
import cv2
import os
import pickle
from tempfile import NamedTemporaryFile
from descriptor import glcm_beta, bitdesc_beta, haralick_feat_beta, features_extraction_concat
from distances import retrieve_similar_image
import glob

PCA_FILENAME = 'Models/pca.pkl'
SCALER_FILENAME = 'Models/scaler.pkl'
LABEL_ENCODER_FILENAME = 'Models/label_encoder.pkl'



def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_signatures(descriptor_type):
    if descriptor_type == "GLCM":
        return np.load('glcm_signatures.npy', allow_pickle=True)
    elif descriptor_type == "BIT":
        return np.load('bit_signatures.npy', allow_pickle=True)
    elif descriptor_type == "Haralick":
        return np.load('haralick_signatures.npy', allow_pickle=True)
    else:
        return None

def get_images_for_label(labels, base_path='Projet1_Dataset'):
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for label in labels:
            if label in dirs:
                label_path = os.path.join(root, label)
                for file in glob.glob(os.path.join(label_path, '*.*')):
                    image_paths.append(file)
    return image_paths

def display_images(image_paths, num_images):
    cols = st.columns(3)
    for i in range(min(num_images, len(image_paths))):
        col = cols[i % 3]
        img_path = image_paths[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col.image(img, caption=os.path.basename(img_path), use_column_width=True)

def main():
    st.set_page_config(page_title='Feature Extraction', page_icon='🔍', layout="wide")
    st.title('🔍 Feature Extraction')

    st.markdown(
        """
        <style>
        div.stButton > button {
            width: 100%;
            margin: 5px 0;
        }
        div.stButton > button:hover {
            background-color: #4B8BBE;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    class_list = ['iris-setosa', 'iris-versicolour', 'fire', 'nofire']

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Traditional CBIR'):
            st.session_state.active_tab = "Traditional CBIR"
    with col2:
        if st.button('Advanced CBIR'):
            st.session_state.active_tab = "Advanced CBIR"
    with col3:
        if st.button('Multimodal CBIR'):
            st.session_state.active_tab = "Multimodal CBIR"

    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = "Traditional CBIR"

    if st.session_state['active_tab'] == "Traditional CBIR":
        st.sidebar.header("🛠️ Descriptors & Distances")
        descriptor_options = ["GLCM", "BIT", "Haralick"]
        selected_descriptor = st.sidebar.radio("Descriptors", descriptor_options)

        distance_options = ["Manhattan", "Euclidean", "Chebyshev", "Canberra"]
        selected_distance = st.sidebar.radio("Distances", distance_options)

        max_distance = st.sidebar.number_input("Distance maximale", min_value=0.0, value=100.0)

    elif st.session_state['active_tab'] == "Advanced CBIR":
        st.sidebar.header("🧠 Models")
        model_paths = {
            'GradientBoostingClassifier': 'Models/GradientBoostingClassifier.pkl',
            'KNN': 'Models/KNN.pkl',
            'Naive Bayes': 'Models/Naive Bayes.pkl',
            'Decision Tree': 'Models/Decision Tree.pkl',
            'SVM': 'Models/SVM.pkl',
            'Random Forest': 'Models/Random Forest.pkl',
            'LogisticRegression': 'Models/LogisticRegression.pkl'
        }
    
        models = {}
        for name, path in model_paths.items():
            with open(path, 'rb') as file:
                models[name] = pickle.load(file)
        selected_model = st.sidebar.selectbox("Veuillez choisir un modèle:", list(models.keys()))
        model = models[selected_model]
        pca = load_model(PCA_FILENAME)
        scaler = load_model(SCALER_FILENAME)
        label_encoder = load_model(LABEL_ENCODER_FILENAME)

    if st.session_state['active_tab'] == "Traditional CBIR":
        st.write("## Traditional CBIR")
        
        signatures = load_signatures(selected_descriptor)
        if signatures is not None:
            total_images = len(signatures)

            uploaded_file = st.file_uploader("Veuillez téléverser votre image:", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_image_path = temp_file.name

                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(uploaded_file, caption='Image téléversée.', use_column_width=False)
            
                with col2:
                    st.write(f"Descripteur sélectionné : {selected_descriptor}")
                    st.write(f"Distance sélectionnée : {selected_distance}")
                    st.write(f"Distance maximale : {max_distance}")

                features = None
                if selected_descriptor == "GLCM":
                    features = glcm_beta(temp_image_path)[:6]
                elif selected_descriptor == "BIT":
                    features = bitdesc_beta(temp_image_path)[:14]
                elif selected_descriptor == "Haralick":
                    features = haralick_feat_beta(temp_image_path)

                sorted_results = retrieve_similar_image(signatures, features, selected_distance.lower(), total_images)
                filtered_results = [result for result in sorted_results if result[1] <= max_distance]

                st.sidebar.write(f"Nombre total d'images similaires trouvées : {len(filtered_results)}")
                num_res_options = list(range(1, len(filtered_results) + 1))
                selected_num_res = st.sidebar.selectbox("Num Res", num_res_options)

                st.write(f"Top {selected_num_res} résultats les plus proches :")
                cols = st.columns(3)
                for i, result in enumerate(filtered_results[:selected_num_res]):
                    col = cols[i % 3]
                    col.write(f"Label : {result[2]}")
                    similar_image = cv2.imread(result[0])
                    similar_image = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)
                    col.image(similar_image, caption=f"Similar Image (Distance: {result[1]:.6f})", use_column_width=True)

                os.remove(temp_image_path)

    elif st.session_state['active_tab'] == "Advanced CBIR":
        st.write("## Advanced CBIR")
        uploaded_file = st.file_uploader("Veuillez téléverser votre image:", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_image_path = temp_file.name

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(uploaded_file, caption='Image téléversée.', use_column_width=False)
        
            with col2:
                st.write(f"Modèle sélectionné : {selected_model}")

            features = features_extraction_concat(temp_image_path)
            features = np.array(features).reshape(1, -1)

            features_scaled = scaler.transform(features)
            features_pca = pca.transform(features_scaled)
            Y_pred = model.predict(features_pca)
            class_name = label_encoder.inverse_transform(Y_pred)
            st.sidebar.markdown(
    f"<p style=' font-weight:bold;'>Prediction: {class_name[0]}</p>",
    unsafe_allow_html=True
)


            image_paths = get_images_for_label(class_name[0])
            st.sidebar.write(f"Nombre total d'images similaires: {len(image_paths)}")
            
            if image_paths:
                num_images_to_show = st.sidebar.slider(
                    "Nombre d'images à afficher: ",
                    min_value=1,
                    max_value=len(image_paths))
                
                st.write(f"Top {num_images_to_show} d'images similaires: ")
                cols = st.columns(3)
                for i, img_path in enumerate(image_paths[:num_images_to_show]):
                    col = cols[i % 3]
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    col.image(img, caption=os.path.basename(img_path), use_column_width=True)
            else:
                st.write(f"Aucune image similaire trouvée")

            os.remove(temp_image_path)
    elif st.session_state['active_tab'] == "Multimodal CBIR":
        st.write("## Multimodal CBIR")

        user_input = st.text_input("Entrez votre demande :")

        if user_input:
            try:
                user_list = user_input.split(' ')
                num = []
                for i in (user_list):
                    try: 
                        res = int(i)
                        num.append(res)
                    except:
                        pass
    
               
                num_photos = num[0]                
                st.write(num_photos)


                labels = user_input.split(" ")
                labels = [label for label in labels if label in class_list]
                # label = [lbl.lower() for lbl in labels if lbl in class_list] 
                st.write(labels)
                
                image_paths = get_images_for_label(labels)
                
                if image_paths:
                    st.write(f"Nombre total d'images trouvées : {len(image_paths)}")
                    
                    display_images(image_paths, num_photos)
                else:
                    st.write(f"Aucune image trouvée pour  '{labels}'")
            except (IndexError, ValueError):
                st.write("Erreur dans la demande.")


if __name__ == '__main__':
    main()
