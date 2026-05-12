import os
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
import time
from ai_model import BiometricAuthModel

# Wczytanie zmiennych z pliku .env (jeśli istnieje)
load_dotenv()

app = Flask(__name__)

# Pobiera SECRET_KEY z pliku .env. Jeśli go nie znajdzie, używa wartości awaryjnej.
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'zapasowy_sekretny_klucz_lokalny')
socketio = SocketIO(app, async_mode='eventlet')

# Pobiera link do bazy z pliku .env. Domyślnie używa localhosta.
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['biometria_db']
collection = db['sesje_uzytkownikow']

# Trasa dla strony logowania
@app.route('/')
def login():
    return render_template('login.html')

# Trasa dla strony głównej (po zalogowaniu)
@app.route('/dashboard')
def dashboard():
    username = request.cookies.get('user_id')
    if not username:
        return redirect(url_for('login'))  
    return render_template('index.html', username=username)

# Trasa dla wylogowania
@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    response.set_cookie('user_id', '', expires=0)
    return response

# Trasa do zapisywania danych biometrycznych
@app.route('/api/biometrics', methods=['POST'])
def save_biometrics():
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
    
    data = request.json
    if data:
        record = {
            'username': username,
            'events': data.get('events', [])
        }
        collection.insert_one(record)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Puste dane'}), 400

# Trasa dla strony rozpoznawania
@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

# Trasa do rozpoznawania (bez zapisu do bazy)
@app.route('/api/recognize', methods=['POST'])
def recognize_biometrics():
    data = request.json
    if data:
        username = request.cookies.get('user_id')
        if not username:
            return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
        
        # Inicjalizacja modelu dla użytkownika
        model = BiometricAuthModel(username)
        
        if not model.is_trained:
             return jsonify({'status': 'error', 'message': 'Model nie jest jeszcze wytrenowany dla tego użytkownika'}), 400

        # Przekazanie zgromadzonych do weryfikacji danych
        result = model.predict(data.get('events', []))
        
        if result is None:
             return jsonify({'status': 'error', 'message': 'Brak wystarczających punktów trajektorii w przesłanych danych'}), 400

        return jsonify({
            'status': 'success',
            'recognized_user': username,
            'is_correct': result['is_correct_user'],
            'confidence': result['confidence']
        })

    return jsonify({'status': 'error', 'message': 'Puste dane'}), 400

@app.route('/api/train', methods=['POST'])
def train_model():
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
        
    model = BiometricAuthModel(username)
    success = model.train_model()
    
    if success:
        return jsonify({'status': 'success', 'message': 'Model został pomyślnie wytrenowany i zapisany.'})
    else:
        return jsonify({'status': 'error', 'message': 'Za mało zgromadzonych danych aby wytrenować model.'}), 400

@app.route('/api/evaluate', methods=['GET'])
def evaluate_user_model():
    username = request.args.get('username')
    if not username:
        username = request.cookies.get('user_id')
        
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak użytkownika do ewaluacji'}), 400
        
    model = BiometricAuthModel(username)
    results = model.evaluate_performance()
    
    if results:
        return jsonify({'status': 'success', 'evaluation': results})
    else:
        return jsonify({'status': 'error', 'message': 'Ewaluacja nie powiodła się. Za mało danych.'}), 400

if __name__ == '__main__':
    # Uruchomienie serwera z obsługą WebSockets
    socketio.run(app, debug=True, port=5000)