# MedAI Virtual Hospital Assistant Chatbot (Final Unique Version)
# Run: python medai_chatbot_final_unique.py

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = [
    {"q": "What services do you offer?",
     "a": "MedAI provides: (1) appointment scheduling, (2) virtual consultations, (3) symptom guidance (non-diagnostic), (4) medication reminders, and (5) health education and lifestyle tips."},
    {"q": "Do you support diabetes risk screening?",
     "a": "Yes. MedAI supports early-risk screening through validated predictive models used by clinicians. The screening is not a final diagnosis; it flags potential risk and recommends follow-up with a healthcare professional."},
    {"q": "How do you protect privacy?",
     "a": "MedAI applies access control, data minimization, encryption in transit/at rest, and audit logging. Only authorized staff can access patient records, and data is used for care and service improvement under governance controls."},
    {"q": "Can I cancel or reschedule an appointment?",
     "a": "Yes. You can cancel or reschedule by providing your appointment reference, or by telling me the doctor name and date/time you booked."},
    {"q": "Do you provide emergency care?",
     "a": "MedAI is not an emergency service. If there is a medical emergency, contact local emergency services immediately."},
]

kb_questions = [item["q"] for item in KB]
vectorizer = TfidfVectorizer(stop_words="english")
kb_matrix = vectorizer.fit_transform(kb_questions)

def kb_answer(user_text: str, min_score: float = 0.22) -> Optional[Tuple[str, float]]:
    v = vectorizer.transform([user_text])
    sims = cosine_similarity(v, kb_matrix).ravel()
    idx = int(sims.argmax())
    score = float(sims[idx])
    if score >= min_score:
        return KB[idx]["a"], score
    return None

EMERGENCY_PATTERNS = [
    r"chest pain",
    r"difficulty breathing|shortness of breath",
    r"faint(ing)?|passed out",
    r"severe bleeding|bleeding a lot",
    r"stroke|face droop|slurred speech|one side weakness",
    r"suicidal|harm myself|kill myself",
]

def is_emergency(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in EMERGENCY_PATTERNS)

def emergency_message() -> str:
    return (
        "⚠️ This may be urgent. I can’t provide emergency medical help. "
        "Please contact local emergency services immediately or go to the nearest emergency department. "
        "If you’re not alone, ask someone nearby to help you."
    )

INTENTS = {
    "greet": ["hi", "hello", "hey", "good morning", "good evening"],
    "help": ["help", "what can you do", "menu", "options"],
    "services": ["services", "what do you offer", "features"],
    "book": ["book", "appointment", "schedule", "reserve", "consultation"],
    "cancel": ["cancel", "reschedule", "change appointment"],
    "reminder": ["remind", "reminder", "medicine", "medication"],
    "symptoms": ["symptom", "pain", "fever", "cough", "headache", "dizzy", "nausea"],
    "privacy": ["privacy", "data", "confidential", "security"],
    "exit": ["exit", "quit", "bye", "goodbye"],
}

def detect_intent(text: str) -> str:
    t = text.lower().strip()
    if is_emergency(t):
        return "emergency"
    scores = {k: 0 for k in INTENTS}
    for intent, kws in INTENTS.items():
        for kw in kws:
            if kw in t:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"

def parse_datetime(user_text: str):
    return dateparser.parse(
        user_text,
        settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
    )

def normalize_doctor_name(text: str) -> Optional[str]:
    t = text.strip()
    if len(t) < 2:
        return None
    if not re.search(r"[a-zA-Z]", t):
        return None
    if not t.lower().startswith("dr"):
        return "Dr " + t
    return t

def menu() -> str:
    return (
        "Here’s what I can help with:\n"
        "1) Book an appointment\n"
        "2) Reschedule/cancel an appointment\n"
        "3) Symptom guidance (non-diagnostic)\n"
        "4) Medication reminders\n"
        "5) MedAI services & privacy FAQs\n"
        "Type: 'book', 'symptoms', 'reminder', 'services', or 'privacy'."
    )

@dataclass
class SessionState:
    mode: str = "idle"
    booking: Dict[str, Optional[str]] = field(default_factory=lambda: {"doctor": None, "datetime": None, "reason": None})
    reminder: Dict[str, Optional[str]] = field(default_factory=lambda: {"medication": None, "time": None})

def bot_reply(user_text: str, state: SessionState) -> str:
    user_text = str(user_text).strip()
    if not user_text:
        return "Please type a message (e.g., 'book appointment', 'services')."

    if is_emergency(user_text):
        state.mode = "idle"
        return emergency_message()

    if state.mode == "booking":
        b = state.booking
        if b["doctor"] is None:
            doc = normalize_doctor_name(user_text)
            if doc is None:
                return "Please provide the doctor’s name (e.g., Dr Ahmed or Ahmed)."
            b["doctor"] = doc
            return f"Noted ✅ {doc}. What date/time do you prefer? (e.g., 'next Monday 10am')"

        if b["datetime"] is None:
            dt = parse_datetime(user_text)
            if dt is None:
                return "I couldn’t understand the date/time. Try like: 'tomorrow 5pm' or '12 March 9:30am'."
            b["datetime"] = dt.isoformat(sep=" ", timespec="minutes")
            return "Thanks. What is the reason for visit? (e.g., follow-up, checkup, diabetes screening)"

        if b["reason"] is None:
            b["reason"] = user_text[:80]
            state.mode = "idle"
            ref = f"MED-{abs(hash((b['doctor'], b['datetime'], b['reason']))) % 10_000_000:07d}"
            return (
                "✅ Appointment booked\n"
                f"• Doctor: {b['doctor']}\n"
                f"• Date/Time: {b['datetime']}\n"
                f"• Reason: {b['reason']}\n"
                f"• Reference: {ref}\n\n"
                "If you want to reschedule/cancel, type: 'reschedule' or 'cancel'."
            )

    if state.mode == "reminder":
        r = state.reminder
        if r["medication"] is None:
            r["medication"] = user_text[:60]
            return "Got it. When should I remind you? (e.g., 'tomorrow 8am')"
        if r["time"] is None:
            dt = parse_datetime(user_text)
            if dt is None:
                return "I couldn’t parse the time. Try: 'today 9pm' or 'tomorrow morning 8am'."
            r["time"] = dt.isoformat(sep=" ", timespec="minutes")
            state.mode = "idle"
            return (
                "✅ Reminder created (demo)\n"
                f"• Medication: {r['medication']}\n"
                f"• Time: {r['time']}\n\n"
                "(In a real system, this would be saved to the MedAI scheduling service.)"
            )

    intent = detect_intent(user_text)

    if intent == "greet":
        return "🏥 Hello! Welcome to MedAI.\n\n" + menu()

    if intent == "help":
        return menu()

    if intent == "services":
        return KB[0]["a"]

    if intent == "privacy":
        return KB[2]["a"]

    if intent == "book":
        state.mode = "booking"
        state.booking = {"doctor": None, "datetime": None, "reason": None}
        return "Sure ✅ Let’s book your appointment. Which doctor would you like to see?"

    if intent == "reminder":
        state.mode = "reminder"
        state.reminder = {"medication": None, "time": None}
        return "Sure ✅ What medication should I remind you about?"

    if intent == "cancel":
        return "To cancel/reschedule, share your appointment reference (MED-xxxxxxx) or tell me the doctor name + date/time."

    if intent == "symptoms":
        return (
            "I can help with *general guidance* (not diagnosis).\n"
            "1) Describe your symptoms and duration.\n"
            "2) If severe symptoms (chest pain, trouble breathing, fainting, severe bleeding), seek urgent care immediately."
        )

    if intent == "exit":
        return "Goodbye 👋 Stay safe and take care."

    faq = kb_answer(user_text)
    if faq:
        return f"(FAQ match score={faq[1]:.2f})\n" + faq[0]

    return "I’m not fully sure I understood. Try: 'book appointment', 'services', 'privacy', 'symptoms', or 'reminder'."


