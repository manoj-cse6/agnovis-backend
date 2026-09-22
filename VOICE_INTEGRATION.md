# Browser Voice Integration Guide — SIH 26131

This guide documents the lightweight, zero-cost, browser-native voice integration for the SIH 26131 backend.

## Architecture

```
[Farmer's Voice] 
       │
       ▼ (Microphone Input)
[Browser Web Speech API: SpeechRecognition (STT)]
       │
       ▼ (Natural Language Text / Any Indian Language)
[POST /chat endpoint (FastAPI Backend)]
       │
       ▼ (Official google-genai SDK)
[Gemini AI Assistant (Configurable via GEMINI_MODEL)]
       │
       ▼ (Text Response in Farmer's Language)
[Browser Web Speech API: SpeechSynthesis (TTS)]
       │
       ▼ (Spoken Audio Output)
[Farmer Hears Advice]
```

### Why This Architecture?
1. **Zero External Paid Services**: Uses standard W3C Web Speech APIs (`SpeechRecognition` / `webkitSpeechRecognition` and `SpeechSynthesis`) supported natively in Chrome, Edge, Safari, and Android browsers.
2. **Multilingual Support**: Supports multilingual speech input and speech output across Indian languages (Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, English, etc.) by setting the `lang` parameter (e.g., `te-IN`, `hi-IN`, `ta-IN`, `en-IN`).
3. **Backend Stability**: The FastAPI backend remains clean, robust, and text-based (`POST /chat`), avoiding heavy binary audio processing or destabilizing dependencies on the server.

---

## Minimal Frontend JavaScript Example

```javascript
// ============================================================
// 1. Initialize Browser Speech-to-Text (STT)
// ============================================================
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
    alert("Web Speech API is not supported in this browser. Please use Chrome or Edge.");
}

const recognizer = new SpeechRecognition();
recognizer.continuous = false;
recognizer.interimResults = false;
// Set language: e.g., 'te-IN' (Telugu), 'hi-IN' (Hindi), 'en-IN' (Indian English)
recognizer.lang = 'en-IN'; 

// Triggered when user finishes speaking
recognizer.onresult = async (event) => {
    const transcript = event.results[0][0].transcript;
    console.log("Transcribed speech:", transcript);
    
    // Send transcribed text to backend /chat
    await sendToChat(transcript);
};

recognizer.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
};

// Function to start listening (e.g. attached to a microphone button)
function startListening() {
    recognizer.start();
}

// ============================================================
// 2. Send Transcribed Text to Backend POST /chat
// ============================================================
async function sendToChat(userMessage, analysisContext = null) {
    const payload = {
        message: userMessage,
        analysis_context: analysisContext // optional crop predict result
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Chat request failed");
        }

        const data = await response.json();
        console.log("Gemini reply:", data.reply);

        // Speak the reply to the farmer
        speakText(data.reply);
    } catch (error) {
        console.error("Error communicating with /chat:", error);
    }
}

// ============================================================
// 3. Browser Text-to-Speech (TTS)
// ============================================================
function speakText(text, lang = 'en-IN') {
    if (!('speechSynthesis' in window)) {
        console.warn("Speech synthesis not supported in this browser.");
        return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.95; // Slightly slower pace for clear comprehension

    // Select an appropriate voice if available
    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(v => v.lang.startsWith(lang.slice(0, 2)));
    if (matchingVoice) {
        utterance.voice = matchingVoice;
    }

    window.speechSynthesis.speak(utterance);
}
```

---

## Standalone Demo

A complete, working single-file HTML demonstration is provided in `voice_demo.html`.
To use it:
1. Start the FastAPI backend: `uvicorn main:app --reload`
2. Open `voice_demo.html` in Google Chrome or Microsoft Edge.
3. Click **"Start Speaking"**, ask an agricultural question in your preferred language, and listen to Gemini's spoken response!
