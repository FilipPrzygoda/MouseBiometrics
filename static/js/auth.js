// Funkcja logowania
if (document.getElementById('loginForm')) {
    document.getElementById('loginForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value.trim();
        
        if (username) {
            // Ustawienie ciasteczka 'user_id' na 24 godziny
            const date = new Date();
            date.setTime(date.getTime() + (24 * 60 * 60 * 1000));
            document.cookie = `user_id=${username}; expires=${date.toUTCString()}; path=/`;
            
            // Przekierowanie na stronę z zadaniami (dashboard)
            window.location.href = '/dashboard';
        }
    });
}

function handleRecognitionMode() {
    const username = document.getElementById('username').value.trim();
    
    if (username) {
        // Ustawienie ciasteczka 'user_id' na 24 godziny
        const date = new Date();
        date.setTime(date.getTime() + (24 * 60 * 60 * 1000));
        document.cookie = `user_id=${username}; expires=${date.toUTCString()}; path=/`;
        
        // Przekierowanie na stronę z modelem AI (recognition)
        window.location.href = '/recognition';
    } else {
        alert('Podaj nazwę użytkownika przed przejściem do trybu rozpoznawania.');
    }
}

// Funkcja wylogowania
function logout() {
    // Usuwanie ciasteczka user_id
    document.cookie = 'user_id=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    
    // Przekierowanie na stronę logowania
    window.location.href = '/logout';
}