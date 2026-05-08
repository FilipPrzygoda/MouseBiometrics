from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sekretny_klucz_projektu_fpdl'
socketio = SocketIO(app, async_mode='eventlet')

# Konfiguracja MongoDB
client = MongoClient('mongodb://localhost:27017/')
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
        # Tutaj można dodać logikę rozpoznawania na podstawie danych biometrycznych
        # na razie zostawiamy puste, bo to będzie część AI
        return jsonify({'status': 'success', 'message': 'Dane otrzymane, ale rozpoznawanie nie jest jeszcze zaimplementowane'})
    return jsonify({'status': 'error', 'message': 'Puste dane'}), 400
if __name__ == '__main__':
    # Uruchomienie serwera z obsługą WebSockets
    socketio.run(app, debug=True, port=5000)
