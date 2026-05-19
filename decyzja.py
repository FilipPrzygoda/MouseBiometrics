"""
Moduł decyzyjny - podejmowanie decyzji przy użyciu wytrenowanego modelu
i danych z całej sesji po normalizacji.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from normalizacja import BiometricNormalizer


class BiometricDecision:
    """
    Klasa odpowiadająca za podejmowanie decyzji autentykacji na podstawie wytrenowanego modelu.
    Używa aggregacji danych z całej sesji (majority voting >= 50%).
    """
    
    def __init__(self, username):
        self.username = username
        self.normalizer = BiometricNormalizer(username)
        
        # Katalog dla modeli
        self.models_dir = 'models'
        self.model_path = os.path.join(self.models_dir, f'model_{self.username}.pkl')
        self.scaler_path = os.path.join(self.models_dir, f'scaler_{self.username}.pkl')
        
        # Model i skaler
        self.model = None
        self.scaler = None
        
        # Załaduj model
        self.load_model()
    
    def load_model(self):
        """Ładuje model i skaler z pliku."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Brak wytrenowanego modelu dla użytkownika {self.username}. "
                f"Upewnij się, że model został wytrenowany przy użyciu BiometricTrainer."
            )
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print(f"[DECYZJA] Model załadowany dla: {self.username}")
        except Exception as e:
            raise RuntimeError(f"Błąd podczas ładowania modelu: {e}")
    
    def _extract_features_from_session(self, events):
        """
        Ekstrakcja cech z sesji (bez is_target_user i session_id).
        
        Args:
            events: Lista zdarzeń myszy
            
        Returns:
            DataFrame z cechami (bez session_id i is_target_user)
        """
        session_data = {'events': events}
        df_features = self.normalizer.normalize_features_for_prediction(session_data)
        
        if df_features.empty:
            return None
        
        return df_features
    
    def predict(self, events):
        """
        Podejmuje decyzję na podstawie przesłanych zdarzeń (całej sesji).
        
        Agreguje predykcje z każdego ruchu i stosuje majority voting (>= 50%).
        
        Args:
            events: Lista zdarzeń myszy z przeglądarki
            
        Returns:
            Dict zawierający:
            - is_correct_user: Bool (True = autentyczny użytkownik)
            - confidence: Float (procent głosów za autentycznym użytkownikiem)
        """
        if self.model is None or self.scaler is None:
            return None
        
        # Ekstrakcja cech
        df_features = self._extract_features_from_session(events)
        
        if df_features is None or df_features.empty:
            return None
        
        # Skalowanie
        X_scaled = self.scaler.transform(df_features)
        
        # Predykcja dla każdego ruchu
        predictions = self.model.predict(X_scaled)
        
        # Majority voting (średnia predykcji >= 0.5 = autentyczny)
        confidence = np.mean(predictions)
        is_correct_user = confidence >= 0.5
        
        # Logging konsoli
        decision_text = "✅ AUTENTYCZNY" if is_correct_user else "❌ ANOMALIA/INNY"
        print(f"\n[DECYZJA] Użytkownik: {self.username}")
        print(f"[DECYZJA] Liczba ruchów w sesji: {len(predictions)}")
        print(f"[DECYZJA] Predykcja: {decision_text}")
        print(f"[DECYZJA] Pewność modelu: {confidence:.4f} ({confidence*100:.1f}%)")
        print()
        
        return {
            'is_correct_user': bool(is_correct_user),
            'confidence': float(confidence)
        }
    
    def predict_with_details(self, events):
        """
        Podejmuje decyzję z dodatkowymi szczegółami.
        
        Returns:
            Dict zawierający:
            - is_correct_user: Bool
            - confidence: Float
            - num_movements: Int (liczba ruchów w sesji)
            - event_predictions: List (predykcja dla każdego ruchu)
        """
        if self.model is None or self.scaler is None:
            return None
        
        df_features = self._extract_features_from_session(events)
        
        if df_features is None or df_features.empty:
            return None
        
        X_scaled = self.scaler.transform(df_features)
        predictions = self.model.predict(X_scaled)
        
        confidence = np.mean(predictions)
        is_correct_user = confidence >= 0.5
        
        # Logging konsoli
        decision_text = "✅ AUTENTYCZNY" if is_correct_user else "❌ ANOMALIA/INNY"
        print(f"\n[DECYZJA-SZCZEGÓŁY] Użytkownik: {self.username}")
        print(f"[DECYZJA-SZCZEGÓŁY] Liczba ruchów w sesji: {len(predictions)}")
        print(f"[DECYZJA-SZCZEGÓŁY] Predykcje per ruch: {[int(p) for p in predictions]}")
        print(f"[DECYZJA-SZCZEGÓŁY] Predykcja: {decision_text}")
        print(f"[DECYZJA-SZCZEGÓŁY] Pewność modelu: {confidence:.4f} ({confidence*100:.1f}%)")
        print()
        
        return {
            'is_correct_user': bool(is_correct_user),
            'confidence': float(confidence),
            'num_movements': len(predictions),
            'event_predictions': [int(p) for p in predictions]
        }
