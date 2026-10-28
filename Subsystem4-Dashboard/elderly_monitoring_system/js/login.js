import { getAuth, onAuthStateChanged, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";
import { firebaseApp } from "./firebase-config.js";

const auth = getAuth(firebaseApp);
const form = document.getElementById("loginForm");
const errorMessage = document.getElementById("loginError");
const submitButton = document.getElementById("loginButton");

onAuthStateChanged(auth, (user) => {
    if (user) {
        window.location.replace("dashboard.html");
    }
}, (error) => {
    showError(`Authentication status could not be checked: ${error.message}`);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorMessage.classList.add("d-none");
    submitButton.disabled = true;
    submitButton.textContent = "Signing in…";

    try {
        await signInWithEmailAndPassword(
            auth,
            document.getElementById("loginEmail").value.trim(),
            document.getElementById("loginPassword").value
        );
    } catch (error) {
        showError(error.message);
        submitButton.disabled = false;
        submitButton.textContent = "Sign in";
    }
});

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("d-none");
}
