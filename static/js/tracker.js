window.BiometricTracker = (function() {
    let biometricData = [];
    let isTracking = false;
    let currentDiamondLeft = null;
    let currentDiamondTop = null;
    let currentScore = 0;

    function logMouseEvent(e, eventType) {
        if (!isTracking) return;
        
        biometricData.push({
            type: eventType || e.type,
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now(), 
            diamondLeft: currentDiamondLeft,
            diamondTop: currentDiamondTop,
            score: currentScore
        });
    }

    // Nasłuchiwanie ruchów myszy
    document.addEventListener('mousemove', function(e) { logMouseEvent(e); });
    document.addEventListener('mousedown', function(e) { logMouseEvent(e); });
    document.addEventListener('mouseup', function(e) { logMouseEvent(e); });

    function sendData() {
        if (biometricData.length === 0) return;
        
        const endpoint = window.TRACKER_API_ENDPOINT || '/api/biometrics';
        const payload = {
            events: biometricData
        };
        
        // Jeśli jesteśmy w trybie rozpoznawania, dodaj zalogowaną nazwę użytkownika
        if (window.currentUser) {
            payload.username = window.currentUser;
        }
        
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Dane odebrane z serwera:', data);
            if (data.recognized_user && window.onUserRecognized) {
                window.onUserRecognized(data.recognized_user, data.confidence, data.is_correct);
            }
        })
        .catch(err => console.error('Błąd zapisu danych:', err));
        
        // Wyczyść po wysłaniu
        biometricData = [];
    }

    return {
        start: function() {
            biometricData = [];
            isTracking = true;
            currentScore = 0;
            currentDiamondLeft = null;
            currentDiamondTop = null;
        },
        stop: function() {
            isTracking = false;
            currentDiamondLeft = null;
            currentDiamondTop = null;
            sendData();
        },
        setDiamondPos: function(pos) {
            if (pos && typeof pos === 'object') {
                currentDiamondLeft = pos.left;
                currentDiamondTop = pos.top;
            }
        },
        setScore: function(score) {
            currentScore = score;
        },
        logDiamondClick: function(e) {
            logMouseEvent(e, 'diamond_click');
        },
    };
})();
