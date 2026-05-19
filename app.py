import os
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
import time
from trening import BiometricTrainer
from decyzja import BiometricDecision

# Wczytanie zmiennych z pliku .env (jeśli istnieje)
load_dotenv()

app = Flask(__name__)

# Pobiera SECRET_KEY z pliku .env. Jeśli go nie znajdzie, używa wartości awaryjnej.
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'zapasowy_sekretny_klucz_lokalny')
socketio = SocketIO(app)

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
        
        try:
            # Inicjalizacja modelu decyzyjnego dla użytkownika
            decision_model = BiometricDecision(username)
            
            # Wykonanie predykcji na bazie przesłanych danych
            result = decision_model.predict(data.get('events', []))
            
            if result is None:
                return jsonify({'status': 'error', 'message': 'Brak wystarczających danych w przesłanej sesji'}), 400

            return jsonify({
                'status': 'success',
                'recognized_user': username,
                'is_correct': result['is_correct_user'],
                'confidence': result['confidence']
            })
        except FileNotFoundError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Błąd predykcji: {str(e)}'}), 500

    return jsonify({'status': 'error', 'message': 'Puste dane'}), 400

@app.route('/api/train', methods=['POST'])
def train_model():
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
        
    try:
        # Inicjalizacja trenera dla użytkownika
        trainer = BiometricTrainer(username, db_uri=MONGO_URI)
        success = trainer.train()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model został pomyślnie wytrenowany i zapisany.'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Za mało zgromadzonych danych aby wytrenować model.'
            }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Błąd treningu: {str(e)}'
        }), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Sprawdza status modelu dla zalogowanego użytkownika."""
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
    
    import os
    models_dir = 'models'
    model_path = os.path.join(models_dir, f'model_{username}.pkl')
    is_trained = os.path.exists(model_path)
    
    return jsonify({
        'status': 'success',
        'username': username,
        'model_trained': is_trained
    })

if __name__ == '__main__':
    # Uruchomienie serwera z obsługą WebSockets
    socketio.run(app, debug=True, port=5000)