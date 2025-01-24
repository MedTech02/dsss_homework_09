
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline

# LLM-Modell laden
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Lade TinyLlama-Modell...")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype="auto",  # Automatische Auswahl des besten Datentyps basierend auf Hardware
    device_map="auto",  # Optimiert die Nutzung von GPU/CPU
)
print("Modell erfolgreich geladen.")

# Start-Kommando
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, I am an AI assistant. How can I help you today?")

# Process-Funktion
async def process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(f"Benutzernachricht: {user_message}")

    # Generierung
    outputs = pipe(user_message, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]

    # Antwort an den Benutzer senden
    print(f"Antwort generiert: {response}")
    await update.message.reply_text(response)

# Hauptfunktion
def main():
    API_TOKEN = ""

    # Anwendung erstellen
    application = Application.builder().token(API_TOKEN).build()

    # Handler hinzufügen
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))

    # Bot starten
    print("Bot läuft...")
    application.run_polling()

if __name__ == "__main__":
    main()
