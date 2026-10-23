import os
from pathlib import Path

PAKET = Path(__file__).resolve().parent.parent
TEXT_DATEI = PAKET / "InputData/Reports.jsonl"
EVENTLOG_DATEI = PAKET / "InputData/Prepared/Prepared_Production_Data.csv"
SPLIT_DATEI = PAKET / "InputData/Splits/manifest.json"
ERGEBNIS_ORDNER = PAKET / "Results"
DETAIL_ORDNER = ERGEBNIS_ORDNER / "Arbeitsdaten"
ARBEITSDATEN_BEHALTEN = False
PAPER_ORDNER = ERGEBNIS_ORDNER / "Paper"
FORTSCHRITT_ANZEIGEN = True
DURCHLAEUFE = 1
EXTRAKTION_SHOTS = [0, 1, 3]
KLASSIFIKATION_SHOTS = [0, 1, 3]

MODELLE = [
    {"name": "llama3.2:1b", "thinking": False},
    {"name": "llama3.2:3b", "thinking": False},
    {"name": "llama3.1:70b", "thinking": False},
    {"name": "gemma3:4b", "thinking": False},
    {"name": "gemma3:12b", "thinking": False},
    {"name": "gemma3:27b", "thinking": False},
    {"name": "gemma4:e4b", "thinking": False},
    {"name": "gemma4:31b", "thinking": False},
    {"name": "gemma4:31b", "thinking": True},
    {"name": "gpt-oss:20b", "thinking": "low"},
    {"name": "gpt-oss:20b", "thinking": "high"},
    {"name": "gpt-oss:120b", "thinking": "medium"},
    {"name": "ornith-1.5:9b", "thinking": False},
    {"name": "nemotron3:33b", "thinking": False},
    {"name": "nemotron-3-super:120b", "thinking": False},
]

EXTRAKTIONS_BASELINES = ["regex", "crf", "crf_time", "ner", "ner_time", "bert_ner", "bert_ner_time"]
KLASSIFIKATIONS_BASELINES = ["tfidf_logreg", "tfidf_svm", "bert"]
BASELINE_TRAININGSANTEILE = [10, 25, 50, 75, 100]
SPLIT_SEED = 42
MODELL_SEED = 8100
BASELINE_SEED_START = 18100
TRAININGS_SEED_START = 1042
VALIDIERUNGS_SEED = 42
NER_MODELL = PAKET / "Models/distilbert-base-uncased"
BERT_MODELL = PAKET / "Models/bert-base-uncased"
NER_GERAET = "auto"
NER_EPOCHEN = 10
NER_BATCH_GROESSE = 4
NER_LERNRATE = 2e-5
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
TIMEOUT_SEKUNDEN = 600
TEMPERATUR = 0.0
KONTEXT_TOKEN = 4096
ANTWORT_TOKEN_EXTRAKTION = 220
ANTWORT_TOKEN_KLASSIFIKATION = 140
ANTWORT_TOKEN_MIT_THINKING = 8192
ANTWORT_TOKEN_MATCHING = 100
MATCHING_BASELINES = ["crf_time", "ner_time", "bert_ner_time"]
MATCHING_QUELLEN = [
    dict(model, shots=shots) for model in MODELLE for shots in EXTRAKTION_SHOTS
] + MATCHING_BASELINES

MATCHING_TRAININGSANTEILE = [100]
MATCHING_QUELL_DURCHLAUF = 1
MATCHING_LLM_MODUS = "gleiches_modell"
BASELINE_MATCHING_MODELL = {"name": "gemma3:12b", "thinking": False}
MATCHING_MODELLE = [{"name": "gemma3:12b", "thinking": False}]
MATCHING_REIHENFOLGE_MISCHEN = False
MATCHING_REIHENFOLGE_SEED = 17
EMBEDDING_MODELL = "mxbai-embed-large"
VERGLEICH = "whole_embedding"
KOSINUS_SCHWELLE = 0.98
MINDESTABSTAND = 0.0
TOP_K = 10
MIN_ATTRIBUTE_DIREKT = 4
NORMALISIERUNG_PASS1 = False
ZEITFENSTER_SEKUNDEN = 60.0
MIT_ORIGINALBERICHT = True
HYBRID_SCHWELLE = 0.8
GEWICHT_AKTIVITAET = GEWICHT_MASCHINE = GEWICHT_DATUM = GEWICHT_UHRZEIT = GEWICHT_WORKER = 1.0
ZEIT_HALBWERT_MINUTEN = 5.0

GENERIERUNG_EREIGNISSE = PAKET / "InputData/Prepared/Error_Event_Candidates.csv"
GENERIERUNG_AUSGABE = PAKET / "InputData/Generated/Reports.jsonl"
GENERIERUNG_SEED = 73
GENERIERUNG_STAPEL = [200, 200, 800]
GENERIERUNG_MODELL = "mistral-medium-3.5:latest"
GENERIERUNG_HARNESS = True
