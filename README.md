# MVP – Chatbot FAQ para Aspirantes y Estudiantes (NLU + Plantillas)

Este MVP implementa el **núcleo NLU** (clasificación de intención) y un **bucle de chat por CLI** con **plantillas de respuesta**.
Intenciones iniciales: **admisiones, becas, inscripciones, plan_estudios, costos**.

## Requisitos
- Python 3.11+
- `pip install -r requirements.txt`

## Estructura
```
.
├── README.md
├── requirements.txt
├── train.py          # Entrena NLU y guarda model.pkl
├── predict.py        # Predicción 1-disparo (muestra intención y confianza)
├── chatbot_cli.py    # Bucle de chat con umbral y fallback
└── templates.json    # Plantillas de respuesta por intención (editable)
```

## Uso rápido

1) Entrena y guarda el modelo:
```bash
python train.py
```

2) Predice una intención:
```bash
python predict.py "¿Cuándo abren inscripciones para nuevo ingreso?"
```
Salida esperada:
```
Intención: inscripciones | confianza ~0.8
Respuesta: [texto desde templates.json]
```

3) Conversa con el bot en CLI:
```bash
python chatbot_cli.py
```
- Escribe tu pregunta y presiona Enter.
- Escribe `salir` para terminar.
- Si la **confianza** < **umbral** (por defecto 0.45), el bot pedirá una **aclaración breve** o mostrará **derivación** a contacto humano.

## Personalización
- Se puede editar `templates.json` para actualizar textos y enlaces oficiales de cualquier fuente, ya que son de prueba y esta pensado para un campus de una universidad pero se puede adaptar a lo que sea.
- Se puede ajustar el `UMBRAL_CONF` en `chatbot_cli.py` para ser más/menos estricto.
- Ampliar el dataset de prueba que está incrustado en `train.py` (o se puede cargar desde un CSV, JSON, etc...).

## Nota
Este prototipo es local (no web). En el siguiente incremento, se tiene pensado un endpoint Flask.