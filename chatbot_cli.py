import json, joblib, numpy as np

UMBRAL_CONF = 0.45
CONTACTO_HUMANO = "Si prefieres atención humana: atencion@[dominio].edu | Lun-Vie 9:00-16:00"

def softmax(z):
    z = np.array(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def cargar_modelo():
    return joblib.load("model.pkl")

def cargar_templates():
    try:
        with open("templates.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def predecir(model, text):
    clf = model.named_steps['clf']
    classes = clf.classes_
    scores = model.decision_function([text])[0]
    if not hasattr(scores, "__len__"):
        scores = np.array([-scores, scores])
    probs = softmax(scores)
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])

def main():
    print("Chatbot FAQ – CLI (escribe 'salir' para terminar)")
    model = cargar_modelo()
    templates = cargar_templates()

    while True:
        user = input("\nTú: ").strip()
        if not user:
            continue
        if user.lower() in ("salir", "exit", "quit"):
            print("Bot: ¡Hasta luego!")
            break

        intent, conf = predecir(model, user)
        if conf < UMBRAL_CONF:
            print("Bot: No estoy totalmente seguro. ¿Puedes aclarar con una frase breve (p. ej., 'becas calendario' o 'inscripción tardía')?")
            clar = input("Tú: ").strip()
            if clar.lower() in ("salir", "exit", "quit"):
                print("Bot: ¡Hasta luego!")
                break
            intent2, conf2 = predecir(model, clar)
            if conf2 < UMBRAL_CONF:
                print("Bot:", CONTACTO_HUMANO)
                continue
            intent, conf = intent2, conf2

        reply = templates.get(intent, "Lo siento, aún no tengo una plantilla para esta intención.")
        print(f"Bot: ({intent}, conf={conf:.2f}) {reply}")

if __name__ == "__main__":
    main()
