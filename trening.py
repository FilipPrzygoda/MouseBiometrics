"""
Moduł treningu - trenowanie modelu Random Forest i zapis do pliku.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from normalizacja import BiometricNormalizer


class BiometricTrainer:
    """
    Klasa odpowiadająca za trening modelu Random Forest dla autentykacji biometrycznej.
    """
    
    def __init__(self, username, db_uri='mongodb://localhost:27017/', db_name='biometria_db'):
        self.username = username
        self.db_uri = db_uri
        self.db_name = db_name
        self.collection_name = 'sesje_uzytkownikow'
        
        # Inicjalizacja bazy danych
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db[self.collection_name]
        
        # Inicjalizacja normalizatora
        self.normalizer = BiometricNormalizer(username)
        
        # Model i skaler
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        self.is_trained = False
        
        # Katalog dla modeli
        self.models_dir = 'models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        self.model_path = os.path.join(self.models_dir, f'model_{self.username}.pkl')
        self.scaler_path = os.path.join(self.models_dir, f'scaler_{self.username}.pkl')
        
        # Spróbuj załadować istniejący model
        self.load_model()
    
    def _fetch_balanced_sessions(self):
        """
        Pobiera sesje użytkownika docelowego i inne sesje, balansując 50/50.
        
        Returns:
            Tuple (user_sessions, other_sessions)
        """
        import random
        
        user_sessions = list(self.collection.find({'username': self.username}, {'_id': 0}))
        other_sessions = list(self.collection.find({'username': {'$ne': self.username}}, {'_id': 0}))
        
        # Balansoowanie - nie więcej sesji innych niż użytkownika
        if len(user_sessions) < len(other_sessions):
            random.shuffle(other_sessions)
            other_sessions = other_sessions[:len(user_sessions)]
        
        return user_sessions, other_sessions
    
    def train(self):
        """
        Pobiera dane z bazy, trenuje model Random Forest i zapisuje go.
        
        Returns:
            Bool - True jeśli trening się powiódł, False w przeciwnym razie
        """
        print(f"\n[TRENING] Pobieranie danych dla użytkownika: {self.username}...")
        user_sessions, other_sessions = self._fetch_balanced_sessions()
        
        if not user_sessions:
            print(f"[BŁĄD] Brak sesji dla użytkownika {self.username} w bazie.")
            return False
        
        print(f"-> Znaleziono {len(user_sessions)} sesji autentycznych i {len(other_sessions)} sesji anomalii.")
        
        # Przygotowanie dataset
        print("[TRENING] Ekstrakcja cech...")
        df_model = self.normalizer.prepare_dataset_for_training(user_sessions, other_sessions)
        
        if df_model.empty:
            print("[BŁĄD] Brak wystarczających danych do trenowania.")
            return False
        
        print(f"-> Wygenerowano {len(df_model)} rekordów cech z {len(user_sessions) + len(other_sessions)} sesji.")
        
        # Podział na zbiór treningowy i testowy (po sesjach)
        unique_sessions = df_model['session_id'].unique()
        train_sessions, test_sessions = train_test_split(unique_sessions, test_size=0.3, random_state=42)
        
        df_train = df_model[df_model['session_id'].isin(train_sessions)]
        df_test = df_model[df_model['session_id'].isin(test_sessions)]
        
        print(f"\n[PODZIAŁ ZBIORU]")
        print(f"Sesje (Trening / Test): {len(train_sessions)} / {len(test_sessions)}")
        print(f"Cechy (Trening / Test): {len(df_train)} / {len(df_test)}")
        
        # Przygotowanie X i y
        cols_to_drop = ['is_target_user', 'session_id']
        X_train = df_train.drop(cols_to_drop, axis=1)
        y_train = df_train['is_target_user']
        X_test = df_test.drop(cols_to_drop, axis=1)
        y_test = df_test['is_target_user']
        
        # Skalowanie
        print("[TRENING] Skalowanie cech...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Trening
        print("[TRENING] Trening Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Ewaluacja
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"\n[WYNIKI TRENINGU]")
        print(f"Dokładność na zbiorze treningowym: {train_score:.4f}")
        print(f"Dokładność na zbiorze testowym: {test_score:.4f}")
        
        self.is_trained = True
        
        # Zapis modelu
        self.save_model()
        
        return True
    
    def save_model(self):
        """Zapisuje wytrenowany model i skaler do pliku."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"\n[ZAPIS] Model zapisany do: {self.model_path}")
        print(f"[ZAPIS] Skaler zapisany do: {self.scaler_path}")
    
    def load_model(self):
        """Ładuje wytrenowany model i skaler z pliku, jeśli istnieje."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                print(f"[ŁADOWANIE] Model załadowany z: {self.model_path}")
            except Exception as e:
                print(f"[BŁĄD] Nie udało się załadować modelu: {e}")
                self.is_trained = False
        else:
            self.is_trained = False
