// Gra biometryczna: jeden diament pojawia się losowo. Po 10 kliknięciach gra się kończy, a stoper mierzy czas reakcji.
(function() {
    const gameArea = document.getElementById('gameArea');
    const scoreEl = document.getElementById('score');
    const timeEl = document.getElementById('time');
    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (!gameArea) return;

    let score = 0;
    let isGameRunning = false;
    let startTime = null;
    let timerInterval = null;
    const TARGET_SCORE = 10;
    const size = 48; // zgodne z CSS

    function randomPosition(sizeW, sizeH, elemW, elemH) {
        // dodaj buffer 20px od krawędzi, by diament nie był częściowo niewidoczny
        const buffer = 20;
        const x = Math.floor(Math.random() * (sizeW - elemW - 2 * buffer) + buffer);
        const y = Math.floor(Math.random() * (sizeH - elemH - 2 * buffer) + buffer);
        return { x, y };
    }

    function updateTime() {
        if (!startTime) return;
        const now = Date.now();
        const diff = (now - startTime) / 1000;
        timeEl.textContent = diff.toFixed(2); // czas do dwóch miejsc po przecinku
    }

    function spawnDiamond() {
        if (!isGameRunning) return;

        // Usuń istniejący diament
        const existing = gameArea.querySelector('.diamond');
        if (existing) existing.remove();

        const d = document.createElement('div');
        d.className = 'diamond';
        const rect = gameArea.getBoundingClientRect();
        const pos = randomPosition(rect.width, rect.height, size, size);
        d.style.left = pos.x + 'px';
        d.style.top = pos.y + 'px';
        d.title = 'Kliknij!';
        
        const diamondCenterX = Math.floor(rect.left + pos.x + size / 2);
        const diamondCenterY = Math.floor(rect.top + pos.y + size / 2);


        if (window.BiometricTracker) {
            window.BiometricTracker.setDiamondPos({
                left: diamondCenterX,
                top: diamondCenterY
            });
        }

        d.addEventListener('click', function(e) {
            e.stopPropagation();
            score += 1;
            scoreEl.textContent = `${score} / ${TARGET_SCORE}`;
            statusEl.textContent = '⚡ Gra';
            
            if (window.BiometricTracker) {
                window.BiometricTracker.setScore(score);
                window.BiometricTracker.logDiamondClick(e);
            }

            // animacja: powiększenie i zniknięcie
            d.style.transition = 'transform 120ms ease, opacity 200ms';
            d.style.transform = 'scale(1.4) rotate(45deg)';
            d.style.opacity = '0';
            setTimeout(() => d.remove(), 220);
            
            if (score >= TARGET_SCORE) {
                stopGame(true); // Zakończ sukcesem
            } else {
                spawnDiamond(); // Kolejny diament
            }
        });

        gameArea.appendChild(d);
    }

    function startGame() {
        // reset
        score = 0;
        scoreEl.textContent = `${score} / ${TARGET_SCORE}`;
        timeEl.textContent = '0.00';
        statusEl.textContent = '⚡ Gra';

        if (window.BiometricTracker) {
            window.BiometricTracker.start();
            window.BiometricTracker.setScore(0);
        }

        startBtn.disabled = true;
        stopBtn.disabled = false;
        isGameRunning = true;

        startTime = Date.now();
        timerInterval = setInterval(updateTime, 50); // Aktualizujemy czas często, by wyglądało płynnie

        // usuń wszystkie diamenty jeśli pozostały
        gameArea.querySelectorAll('.diamond').forEach(n => n.remove());

        // pojawia się pierwszy diament
        spawnDiamond();
    }

    function stopGame(completed = false) {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        isGameRunning = false;
        
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }

        if (completed) {
            updateTime(); // Precyzyjne wymuszenie ostatecznego czasu po ostatnim kliknięciu
            statusEl.textContent = '✅ Koniec!';
        } else {
            statusEl.textContent = '⏹️ Przerwana';
        }
        
        if (window.BiometricTracker) {
            window.BiometricTracker.stop();
        }

        // usuń wszystkie diamenty
        gameArea.querySelectorAll('.diamond').forEach(n => n.remove());
    }

    startBtn.addEventListener('click', startGame);
    stopBtn.addEventListener('click', stopGame);
})();
