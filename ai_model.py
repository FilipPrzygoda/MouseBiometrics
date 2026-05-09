import numpy as np
import pandas as pd
from pymongo import MongoClient
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class BiometricAuthModel:
    def __init__(self, username, db_uri='mongodb://localhost:27017/', db_name='biometria_db'):
        self.username = username
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db['sesje_uzytkownikow']
        
        self.scaler = StandardScaler()
        # Isolation Forest - contamination to oczekiwany odsetek anomalii w danych uczących. 
        # Ponieważ w danych mamy głównie lub tylko ruchy właściciela, dajemy mały np. 0.05 lub 0.1 (5-10% uważa za anomalię)
        self.model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.is_trained = False
        
        self.models_dir = 'models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        self.model_path = os.path.join(self.models_dir, f'model_{self.username}.pkl')
        self.scaler_path = os.path.join(self.models_dir, f'scaler_{self.username}.pkl')
        self.load_model()
        
    def _extract_features_from_trajectory(self, events):
        """Krok 2 i 3: Ekstrakcja z pojedynczej trajektorii"""
        if len(events) < 3:
            return None
            
        dt_list = []
        dist_list = []
        vel_list = []
        acc_list = []
        angles_list = []
        
        # Krok 2: Ekstrakcja cech chwilowych
        for i in range(1, len(events)):
            p1 = events[i-1]
            p2 = events[i]
            
            # Zmiana czasu
            dt = max((p2.get('timestamp', 0) - p1.get('timestamp', 0)), 1)
            dt_list.append(dt)
            
            # Dystans
            dx = p2.get('x', 0) - p1.get('x', 0)
            dy = p2.get('y', 0) - p1.get('y', 0)
            dist = math.sqrt(dx**2 + dy**2)
            dist_list.append(dist)
            
            # Prędkość
            vel = dist / dt
            vel_list.append(vel)
            
            # Kąt
            angle = math.atan2(dy, dx)
            angles_list.append(angle)
            
        for i in range(1, len(vel_list)):
            acc = (vel_list[i] - vel_list[i-1]) / max(dt_list[i], 1)
            acc_list.append(acc)
            
        if not acc_list:
            acc_list = [0]
            
        # Krok 3: Agregacja do wektora cech
        reaction_time = dt_list[0] if dt_list else 0
        total_time = sum(dt_list)
        v_mean = np.mean(vel_list) if vel_list else 0
        v_max = np.max(vel_list) if vel_list else 0
        a_max = np.max(acc_list) if acc_list else 0
        
        # Wydajność ruchu (Prostoliniowość)
        total_dist = sum(dist_list)
        start_x, start_y = events[0].get('x', 0), events[0].get('y', 0)
        end_x, end_y = events[-1].get('x', 0), events[-1].get('y', 0)
        direct_dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        straightness = (direct_dist / total_dist) if total_dist > 0 else 0
        
        # Trzęsienie dłoni (Jitter)
        jitter = 0
        for i in range(1, len(angles_list)):
            diff = abs(math.degrees(angles_list[i] - angles_list[i-1]))
            # normalizacja do najmniejszego kąta
            if diff > 180:
                diff = 360 - diff
            if diff > 20:
                jitter += 1
                
        # Precyzja
        target_x = events[-1].get('diamondLeft', 0) or 0
        target_y = events[-1].get('diamondTop', 0) or 0
        precision = math.sqrt((end_x - target_x)**2 + (end_y - target_y)**2) if target_x and target_y else 0
        
        return [reaction_time, total_time, v_mean, v_max, a_max, straightness, jitter, precision]

    def _process_raw_data(self, all_events):
        """Krok 1: Grupowanie na trajektorie na podstawie `score`"""
        trajectories = {}
        for event in all_events:
            score = event.get('score')
            if score is None:
                continue
            if score not in trajectories:
                trajectories[score] = []
            trajectories[score].append(event)
            
        features = []
        for score, events in trajectories.items():
            # Filtrujemy tylko ruchy myszką i kliknięcia żeby trajektoria miała sens
            valid_events = [e for e in events if e.get('type') in ['mousemove', 'mousedown', 'diamond_click']]
            feat_vector = self._extract_features_from_trajectory(valid_events)
            if feat_vector is not None:
                features.append(feat_vector)
                
        return features

    def train_model(self):
        """Pobiera dane z bazy, uczy model Isolation Forest i zapisuje go."""
        cursor = self.collection.find({'username': self.username})
        all_features = []
        
        for record in cursor:
            events = record.get('events', [])
            session_features = self._process_raw_data(events)
            all_features.extend(session_features)
            
        if len(all_features) < 10:
            print(f"Za mało danych do wytrenowania modelu dla {self.username} (wymagane min. 10, najlepiej >100). Obecna ilość logów: {len(all_features)}")
            return False
            
        X = np.array(all_features)
        
        # Krok 4: Skalowanie
        X_scaled = self.scaler.fit_transform(X)
        
        # Trening
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Zapis do plików
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model dla {self.username} wytrenowany na {len(all_features)} próbkach (ruchach) i zapisany pomyślnie.")
        
        return True
        
    def predict(self, raw_events):
        """Zwraca wynik dla nowej próbki (z nowej sesji) czy to ten sam użytkownik (1) czy anomalia (-1)."""
        if not self.is_trained:
            print("Model nie jest wytrenowany!")
            return None
            
        features = self._process_raw_data(raw_events)
        if not features:
            print("Brak poprawnych trajektorii w podanych danych.")
            return None
            
        X = np.array(features)
        X_scaled = self.scaler.transform(X)
        
        # Zwraca -1 (anomalia) lub 1 (zgodny użytkownik)
        predictions = self.model.predict(X_scaled)
        
        # Przykładowa agregacja (większość ruchów musi być poprawna)
        valid_percentage = np.sum(predictions == 1) / len(predictions)
        
        is_correct_user = valid_percentage > 0.5 # Wymagamy np. 50% ruchów rozpoznanych jako "normalne"
        
        return {
            'is_correct_user': bool(is_correct_user),
            'confidence': float(valid_percentage),
            'predictions': predictions.tolist()
        }
        
    def load_model(self):
        """Wczytanie wytrenowanego modelu jeśli istnieje"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            print(f"Załadowano wytrenowany model dla {self.username}")
        else:
            self.is_trained = False
