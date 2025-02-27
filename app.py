from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import mysql.connector
import joblib
import re
import json
from random import choice
import pandas as pd

app = Flask(__name__)
app.secret_key = "jivacare"

MEDICAL_DISCLAIMER = "\n\n⚠️ Note: This is not a substitute for professional medical advice. Please consult a doctor for proper diagnosis and treatment."

# Load the trained model and vectorizer
tfidf = joblib.load(
    r'C:\Users\gummi\Documents\Computer Science\Python\Projects\JeevaCare\JIVACARE\models\tfidf_vectorizer.pkl')
model = joblib.load(
    r'C:\Users\gummi\Documents\Computer Science\Python\Projects\JeevaCare\JIVACARE\models\best_lr_model.pkl')

mapping_file_path = r"C:\Users\gummi\Documents\Computer Science\Python\Projects\JeevaCare\JIVACARE\models\mappings.csv"
mappings_df = pd.read_csv(mapping_file_path)

# Load LabelEncoder
label_encoder = joblib.load(
    r'C:\Users\gummi\Documents\Computer Science\Python\Projects\JeevaCare\JIVACARE\models\label_encoder.pkl')


label_mapping = dict(zip(mappings_df["disease"], mappings_df["label"]))

# Reverse the dictionary to get label → disease
label_mapping_inv = {str(v): k for k, v in label_mapping.items()}

# Database connection
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='password',
    database='jivacare'
)

cursor = db.cursor()
# cursor.execute(
#     """CREATE TABLE IF NOT EXISTS users(
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     name VARCHAR(255),
#     age INT,
#     gender VARCHAR(50),
#     allergies TEXT,
#     email VARCHAR(255) UNIQUE,
#     password VARCHAR(255)
#     );
#     """
# )

# Regex patterns for natural conversations
patterns = {
    r"\bhel+o+\b|\bh[iy]+\b|\bhe+y+\b": ["Hello there!", "Hi! How's it going?", "Hey! Nice to see you!", "Hey, what's good?", "Hey!"],
    r"\bhow are you\b|\bwhats? good\b|\bwas+up+\b": ["I'm here and ready to chat with you. How about you?", "I'm just a bot, but I'm happy to talk!", "Just hanging out here. What's on your mind?"],
    r"\bi am ([a-zA-Z\s]+)|\bi (just)? feel ([a-zA-Z\s]+) | i am feeling ([a-zA-Z\s]+)": ["I see you're feeling {0}. Want to talk about it?", "Okay, why do you feel {0}?", "You're feeling {0}, right? Let's work through it together."],
    r"\bwhat is your name\b|\bwhat'?s your name\b|\bwho are you\b|\bwhat are you\b": ["I'm JivaCare🍀, your health assistant!", "You can call me JivaCare🍀, your digital health companion!", "I'm JivaCare🍀, here to help you with your health concerns!"],
    r"\bthank y[ou]+\b|\bthanks+\b|\bthx\b|\bthanku+\b": ["You're welcome! Please ensure you visit a doctor for medical assistance.", "Happy to help! Remember, visit a doctor for medical assistance.", "Anytime! Do visit a doctor if your symptoms persist.", "Ping me when you need me again. Visit a doctor if your symptoms persist."],
    r"\bbye+\b|\bexit\b|\bquit\b|\bsee you\b": ["Goodbye! Hope to talk to you again soon!", "Take care!", "Catch you later!"]
}

# Stopwords to remove
STOP_WORDS = {"i", "and", "the", "is", "a", "very", "think", "im", "an"}

# Tokenize and clean input text


def tokenise(text):
    text = re.sub(r"\bi'm\b|\bim\b", "I am", text)
    text = re.sub(r"\byou're\b|\byoure\b", "you are", text)
    text = re.sub(r"\bcan't\b|\bcant\b", "cannot", text)
    text = re.sub(r"\bwhat's\b|\bwhats\b", "what is", text)
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    words = re.split(r"\s+", text)
    return [word.lower() for word in words if word.lower() not in STOP_WORDS]

# Function to preprocess user input


def preprocess_input(user_input):
    tokens = tokenise(user_input)
    processed_input = " ".join(tokens)
    return tfidf.transform([processed_input])

# Function to predict disease


def predict_disease(user_input):
    processed_input = preprocess_input(user_input)


    # Get the predicted encoded label (0 to 42)
    predicted_encoded_label = model.predict(processed_input)[0]

    # Convert encoded label back to original label (e.g., "1026")
    original_label = label_encoder.inverse_transform(
        [predicted_encoded_label])[0]

    # Retrieve the disease name using the inverse mapping
    disease_name = label_mapping_inv.get(
        str(original_label), "an unknown disease")
    return disease_name


# Function to handle natural conversations


def handle_conversation(user_input):
    user_input = " ".join(tokenise(user_input))  # Tokenize and normalize input

    # Check for vague illness statements
    vague_illness_patterns = [
        r"\bi (feel|am|think i am|think i'm) (sick|unwell|not good|bad|off|ill|weak|dizzy|tired|weird|down|awful)\b",
        r"\bi (am feeling|feel like|have been feeling) (off|not great|under the weather|unwell|bad|down|weak|exhausted|nauseous)\b",
        r"\bi'?m (not feeling well|not well|not okay|feeling off|feeling bad|not myself|kind of sick|sort of sick)\b",
        r"\b(not sure what'?s wrong|i don'?t feel right|i'?m not doing well|i just feel off)\b",
        r"\bsomething feels wrong with me\b",
        r"\bi'?m a bit (off|down|under the weather|out of it)\b",
        r"\bi (may be|might be|could be) sick\b",
        r"\bmy body feels (off|weak|drained|strange|tired|heavy)\b",
        r"\bi think i caught (something|a bug|a cold)\b",
        r"\bi feel like i (have|caught) (something|a virus|a flu|a cold)\b",
    ]

    for pattern in vague_illness_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "I see you're not feeling well. Can you describe your symptoms?"

    # Check general conversation patterns
    for pattern, responses in patterns.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            groups = [g for g in match.groups() if g]  # Extract matched groups
            return choice(responses).format(*groups) if groups else choice(responses)

    return None  # Return None if no regex match is found


# Routes


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name'].strip()
        age = request.form['age'].strip()
        gender = request.form['gender'].strip()
        allergies = request.form['allergies'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()
        confirm_password = request.form['confirm_password'].strip()

        if not (name and age and gender and email and password and confirm_password):
            flash("All fields are required!", "error")
            return render_template("signup.html")

        if not age.isdigit() or int(age) <= 0:
            flash("Enter a valid age!", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return render_template("signup.html")

        cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash("Email already exists!", "error")
            return render_template("signup.html")

        cursor.execute("INSERT INTO users (name, age, gender, allergies, email, password) VALUES (%s, %s, %s, %s, %s, %s)",
                       (name, age, gender, allergies, email, password))
        db.commit()
        flash("Signup successful!", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        cursor.execute(
            "SELECT name FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()

        if user:
            session['name'] = user[0]
            session['email'] = email
            return redirect(url_for('chatbot'))
        else:
            flash("Invalid email or password. Try again.", "error")
            return render_template("login.html")

    return render_template("login.html")


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'name' not in session:
        return redirect(url_for('login'))

    if 'chat_history' not in session:
        session['chat_history'] = []
        welcome_message = (
            "👋 Hi! I am JivaCare🍀, your medical assistant for recommending home remedies. "
            "What seems to be the issue? \n⚠️Please note: I am not a replacement for a doctor. "
            "Always consult a medical professional for serious concerns."
        )
        session['chat_history'].append(("JivaCare", welcome_message))
        session.modified = True

    user_name = session['name']

    if request.method == 'POST':
        user_input = request.form['user_input'].strip()

        # Try regex-based natural conversation first
        response = handle_conversation(user_input)
        if response:
            session['chat_history'].append(("You", user_input))
            session['chat_history'].append(("JivaCare🍀", response))
            session.modified = True
            return render_template('chatbot.html', name=user_name, chat_history=session['chat_history'])
        
        predicted_disease = predict_disease(user_input)

        # If no regex match, predict disease using the ML model
        response_templates = [
            f"I see. It looks like you might have {predicted_disease}. Here's what you can do:",
            f"Hmm... Based on what you're saying, you could have {predicted_disease}. Here are some remedies:",
            f"I think you might be experiencing {predicted_disease}. Here's how you can take care of it:",
            f"It sounds like you might have {predicted_disease}. Try these home remedies:",
            f"I'm not a doctor, but from your symptoms, it seems like {predicted_disease}. You can try this:",
        ]

        if not response:
            response = choice(response_templates)

        # Store conversation in session history
        session['chat_history'].append(("You", user_input))
        session['chat_history'].append(("JivaCare🍀", response))
        session.modified = True  # Ensure session updates

    return render_template('chatbot.html', name=user_name, chat_history=session['chat_history'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
