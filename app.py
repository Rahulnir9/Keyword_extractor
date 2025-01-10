from flask import Flask, render_template, request
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def extract_keywords_tfidf(job_description):
    cleaned_description = clean_text(job_description)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([cleaned_description])
    tfidf_scores = X.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    # Create a DataFrame with words and their TF-IDF scores
    tfidf_df = pd.DataFrame({'word': feature_names, 'tfidf_score': tfidf_scores})
    
    # Sort by TF-IDF score and get top 10 keywords
    return tfidf_df.sort_values(by='tfidf_score', ascending=False).head(10)

def extract_keywords_pos(job_description):
    tokens = word_tokenize(job_description)
    tagged = pos_tag(tokens)
    
    # Filter for nouns and adjectives (NN, NNS, NNPS, JJ)
    keywords = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNPS', 'JJ')]
    
    # Return top 10 keywords based on frequency
    return pd.Series(keywords).value_counts().head(10)

@app.route('/', methods=['GET', 'POST'])
def index():
    keywords = None
    error = None
    if request.method == 'POST':
        job_description = request.form['job_description']
        try:
            # Use either TF-IDF or POS tagging method
            keywords = extract_keywords_tfidf(job_description)  # or use extract_keywords_pos(job_description)
        except Exception as e:
            error = str(e)
    return render_template('index.html', keywords=keywords, error=error)

if __name__ == '__main__':
    app.run(debug=True)