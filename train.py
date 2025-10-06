import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data = [
    ("¿Cuáles son los requisitos de admisión para licenciatura?", "admisiones"),
    ("¿Fechas para el examen de admisión?", "admisiones"),
    ("¿Cómo es el proceso de admisión para extranjeros?", "admisiones"),
    ("¿Documentos necesarios para aplicar?", "admisiones"),
    ("¿Puedo revalidar materias al ingresar?", "admisiones"),
    ("¿Qué becas ofrecen y cuáles son los requisitos?", "becas"),
    ("¿Cuándo abren las convocatorias de becas?", "becas"),
    ("¿Cómo mantengo mi beca activa?", "becas"),
    ("¿Dónde consulto los resultados de becas?", "becas"),
    ("¿Existe beca por promedio?", "becas"),
    ("¿Cuándo abren inscripciones para nuevo ingreso?", "inscripciones"),
    ("¿Cómo completo el proceso de reinscripción?", "inscripciones"),
    ("¿Puedo inscribirme fuera de fecha?", "inscripciones"),
    ("¿Dónde pago mi ficha de inscripción?", "inscripciones"),
    ("¿Cambio de grupo en reinscripción es posible?", "inscripciones"),
    ("¿Cuántos semestres tiene la carrera?", "plan_estudios"),
    ("¿Cuál es la carga de materias por semestre?", "plan_estudios"),
    ("¿Tienen materias optativas y cuáles son?", "plan_estudios"),
    ("¿Puedo adelantar materias del plan?", "plan_estudios"),
    ("¿Hay prácticas profesionales en el plan?", "plan_estudios"),
    ("¿Cuál es el costo por semestre?", "costos"),
    ("¿Cuánto se paga de inscripción y colegiatura?", "costos"),
    ("¿Hay descuentos o convenios de pago?", "costos"),
    ("¿Dónde consulto cuotas actualizadas?", "costos"),
    ("¿Cobran extra por laboratorio?", "costos"),
]

X = [t for t, y in data]
y = [y for t, y in data]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1)),
    ("clf", LinearSVC())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, pred)
print("Precisión (holdout): {:.1f}%".format(acc*100))
print("\nReporte de clasificación:\n")
print(classification_report(y_test, pred))

joblib.dump(pipeline, "model.pkl")
print("\nModelo guardado en model.pkl")
