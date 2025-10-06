import sys, json, joblib
import numpy as np

def softmax(z):
    z = np.array(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR, usa en la terminal >> python predict.py \"<Cualquier pregunta del train.py>\"")
        sys.exit(1)

    text = sys.argv[1]
    model = joblib.load("model.pkl")
    try:
        classes = model.named_steps['clf'].classes_
        scores = model.decision_function([text])[0]
        if not hasattr(scores, "__len__"):
            import numpy as np
            scores = np.array([-scores, scores])
        probs = softmax(scores)
        idx = int(probs.argmax())
        intent = classes[idx]
        conf = float(probs[idx])
    except Exception:
        intent = model.predict([text])[0]
        conf = 0.5

    try:
        with open("templates.json", "r", encoding="utf-8") as f:
            templates = json.load(f)
    except FileNotFoundError:
        templates = {}

    reply = templates.get(intent, "Lo siento, aún no tengo una plantilla para esta intención.")

    print(f"Intención: {intent} | confianza: {conf:.2f}")
    print("Respuesta:", reply)
