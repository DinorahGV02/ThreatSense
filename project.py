#ThreatSense main python file - localhost web application
#importing required libraries
from flask import Flask, request, render_template
import joblib
import tldextract  #for accurately splitting the URLs

app = Flask(__name__)

# Already trained models in the previous steps
#Loading them to this program
email_model = joblib.load('rf_classifier_email.pkl')
url_model = joblib.load('rf_classifier_url.pkl')
email_vectorizer = joblib.load('vectorizer_email.pkl')
url_vectorizer = joblib.load('vectorizer_url.pkl')

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html') #index.html holds the actual website design

# Route for email prediction
@app.route('/predict_email', methods=['POST'])
def predict_email():
    email_text = request.form['email_text']
    email_text_vectorized = email_vectorizer.transform([email_text])
    email_prediction = email_model.predict(email_text_vectorized)
     # Truncating the email_text to 100 characters plus 10 dots for more pleasant visual
    truncated_email_text = (email_text[:100] + '..........') if len(email_text) > 100 else email_text
    return render_template('index.html', email_prediction=email_prediction[0], email_text=truncated_email_text)

# Function that whitelists some of the domains
# We know that this method should not be used since we aim to predict through machine learning
# but to provide a little bit more accurate results, we had to implement the whitelisting
def is_whitelisted(url):
    known_safe_domains = {'youtube.com', 'google.com', 'facebook.com'}
    extracted = tldextract.extract(url)
    full_domain = "{}.{}".format(extracted.domain, extracted.suffix)
    return full_domain in known_safe_domains

# Route for URL prediction
@app.route('/predict_url', methods=['POST'])
def predict_url():
    url_text = request.form['url_text']
    if is_whitelisted(url_text):
        url_prediction = 'benign'
    else:
        url_text_vectorized = url_vectorizer.transform([url_text])
        url_prediction = url_model.predict(url_text_vectorized)
        url_prediction = url_prediction[0]  # Getting decision from prediction array
    return render_template('index.html', url_prediction=url_prediction, url_text=url_text)

if __name__ == '__main__':
    app.run(debug=True)
