import re
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from joblib import load
import streamlit as st

# Téléchargement des ressources nécessaires
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des outils de traitement
lemmatizer = WordNetLemmatizer()
stemmer_fr = SnowballStemmer('french')
stemmer_en = SnowballStemmer('english')
stop_words_fr = stopwords.words('french')
stop_words_en = stopwords.words('english')

# Fonction de prétraitement multilingue
def preprocess_text_multilingual(sen):
    try:
        lang = detect(sen)  # Détection de la langue
    except:
        lang = "unknown"
    
    # Suppression des URLs
    sentence = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', ' ', sen)
    # Suppression des caractères spéciaux, ponctuations et chiffres
    sentence = re.sub('[^a-zA-Zàâçéèêëîïôûùüÿñæœ]', ' ', sentence)
    # Suppression des caractères uniques et espaces multiples
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Conversion en minuscules et suppression des stopwords en fonction de la langue
    if lang == "fr":
        sentence = ' '.join(word.lower() for word in sentence.split() if word not in stop_words_fr)
        sentence = ' '.join(stemmer_fr.stem(word) for word in sentence.split())
    elif lang == "en":
        sentence = ' '.join(word.lower() for word in sentence.split() if word not in stop_words_en)
        sentence = ' '.join(lemmatizer.lemmatize(word) for word in sentence.split())
    else:
        sentence = sentence.lower()  # Cas où la langue n'est pas détectée
    
    return sentence

# Fonction de détection de la langue
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"




# Chargement le modèle et le vectoriseur
model = load("../models/svm_model.joblib")
vectorizer = load("../models/tfidf_vectorizer.joblib")

# Interface utilisateur avec Streamlit
st.title("Détection de Tweets Suspects (FR/EN)")

# Entrée utilisateur
tweet = st.text_area("Entrez un tweet :", "")

if st.button("Analyser"):
    if tweet.strip():
        # Détection la langue
        lang = detect_language(tweet)
        
        if lang not in ["fr", "en"]:
            st.write("Langue non prise en charge. Veuillez entrer un texte en français ou en anglais.")
        else:
            # Prétraitement multilingue
            cleaned_tweet = preprocess_text_multilingual(tweet)
            # Transformation en TF-IDF et prédiction
            X_input = vectorizer.transform([cleaned_tweet])
            prediction = model.predict(X_input)
            result = "Suspect" if prediction[0] == 0 else "Non Suspect"
            st.write(f"**Langue détectée : {'Français' if lang == 'fr' else 'Anglais'}**")
            st.write(f"**Résultat : {result}**")
    else:
        st.write("Veuillez entrer un tweet.")
