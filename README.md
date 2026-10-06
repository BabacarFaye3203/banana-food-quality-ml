# 🍌 Banana Quality ML IoT

Système de **Machine Learning + IoT** pour la surveillance de la qualité post-récolte des bananes.

Le projet vise à construire une chaîne complète permettant de collecter des mesures provenant de capteurs connectés à un **ESP32**, de valider ces mesures et d'utiliser un modèle de Machine Learning pour classifier l'état de conservation des bananes.

Les classes actuellement utilisées sont :

- 🟢 `bon`
- 🟠 `moyen`
- 🔴 `mauvais`

> **État actuel :** le pipeline ML et l'API FastAPI sont fonctionnels avec un dataset IoT synthétique. L'intégration avec les données ESP32 réelles est en cours de validation. Les performances actuelles ne constituent donc pas une validation expérimentale sur des données réelles.

---

## 📋 Table des matières

- [1. Présentation](#1--présentation)
- [2. Objectifs](#2--objectifs)
- [3. État actuel du projet](#3--état-actuel-du-projet)
- [4. Architecture](#4--architecture)
- [5. Technologies](#5--technologies)
- [6. Structure du projet](#6--structure-du-projet)
- [7. Données IoT](#7--données-iot)
- [8. Variables utilisées par le modèle](#8--variables-utilisées-par-le-modèle)
- [9. Génération du dataset](#9--génération-du-dataset)
- [10. Analyse exploratoire](#10--analyse-exploratoire)
- [11. Modélisation](#11--modélisation)
- [12. Résultats expérimentaux](#12--résultats-expérimentaux)
- [13. MLflow](#13--mlflow)
- [14. API FastAPI](#14--api-fastapi)
- [15. Validation des données](#15--validation-des-données)
- [16. Intégration ESP32](#16--intégration-esp32)
- [17. Tests](#17--tests)
- [18. Docker](#18--docker)
- [19. Installation](#19--installation)
- [20. Exécution](#20--exécution)
- [21. Limites](#21--limites)
- [22. Évolution vers un système MLOps](#22--évolution-vers-un-système-mlops)
- [23. Perspectives](#23--perspectives)
- [24. Auteur](#24--auteur)

---

# 1. 📌 Présentation

La conservation post-récolte des bananes dépend de plusieurs facteurs environnementaux et physiologiques.

Dans le cadre de ce projet, un système IoT basé sur un **ESP32** collecte différentes mesures pouvant être utilisées comme indicateurs de l'état de conservation :

- température ;
- humidité ;
- réponse brute du capteur MQ2 ;
- mesure brute du capteur FC28 ;
- détection de liquide/humidité ;
- mouvement ;
- luminosité.

Ces données sont transmises à un service Machine Learning exposé par une API REST.

Le modèle retourne une classe de qualité :

```text
bon
moyen
mauvais
```

Architecture actuelle :

```text
ESP32 / IoT
     │
     ▼
Mesures capteurs
     │
     ▼
Validation
     │
     ▼
Préparation des features
     │
     ▼
Modèle ML
     │
     ▼
FastAPI
     │
     ▼
Prédiction de qualité
```

---

# 2. 🎯 Objectifs

## Objectif principal

Construire un pipeline permettant de :

1. collecter des données provenant de capteurs IoT ;
2. contrôler la qualité des mesures ;
3. transformer les mesures en features ML ;
4. entraîner un modèle de classification ;
5. versionner le modèle avec MLflow ;
6. exposer le modèle via FastAPI ;
7. exécuter l'API dans Docker ;
8. préparer l'intégration avec les données réelles de l'ESP32.

## Objectifs futurs

Le système pourra ensuite évoluer vers :

- la détection d'anomalies ;
- la prédiction du risque de dégradation ;
- l'analyse temporelle ;
- les alertes en temps réel ;
- le monitoring du modèle ;
- le réentraînement automatique.

---

# 3. ✅ État actuel du projet

Les éléments suivants sont actuellement opérationnels ou validés :

| Composant | État |
|---|---|
| Dataset IoT synthétique | ✅ |
| Validation des données | ✅ |
| Feature preparation | ✅ |
| Random Forest | ✅ |
| HistGradientBoosting | ✅ |
| Évaluation du modèle | ✅ |
| MLflow Tracking | ✅ |
| MLflow Model Registry | ✅ |
| FastAPI | ✅ |
| `/health` | ✅ |
| `/predict` | ✅ localement |
| Tests pytest | ✅ |
| Image Docker | ✅ |
| API dans Docker | ✅ |
| Connexion Docker → MLflow | ⚠️ en validation |
| Données ESP32 réelles | 🔄 en cours |
| MQTT | 🔜 prévu |
| Monitoring ML | 🔜 prévu |

---

# 4. 🏗️ Architecture

## Architecture actuelle

```text
                 ┌─────────────────────┐
                 │       ESP32         │
                 │                     │
                 │ Temperature         │
                 │ Humidity            │
                 │ MQ2                 │
                 │ FC28                │
                 │ LDR                 │
                 │ Motion              │
                 └──────────┬──────────┘
                            │
                            │ HTTP
                            ▼
                 ┌─────────────────────┐
                 │      FastAPI        │
                 │                     │
                 │ Validation          │
                 │ Feature preparation │
                 │ Prediction          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       MLflow        │
                 │                     │
                 │ Model Registry      │
                 │ banana_quality_model│
                 └─────────────────────┘
```

## Flux de prédiction

```text
IoT measurement
      │
      ▼
Pydantic validation
      │
      ▼
validate_iot_measurement()
      │
      ▼
Feature selection
      │
      ▼
MLflow Model Registry
      │
      ▼
HistGradientBoosting
      │
      ▼
bon / moyen / mauvais
```

---

# 5. 🛠️ Technologies

## Machine Learning

- Python 3.12
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Random Forest
- HistGradientBoostingClassifier

## API

- FastAPI
- Pydantic
- Uvicorn
- HTTPX

## MLOps

- MLflow 3.16.1
- MLflow Tracking
- MLflow Model Registry

## DevOps

- Docker
- Docker Desktop
- uv

## IoT

- ESP32
- MQ2
- FC28
- LDR
- capteurs de température et d'humidité
- capteur de mouvement

---

# 6. 📁 Structure du projet

```text
model_ml_iot/
│
├── .venv/
│
├── data/
│   ├── raw/
│   │   ├── banana_storage_dataset.csv
│   │   └── banana_iot_model_ready.csv
│   │
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_iot_data_analysis.ipynb
│
├── src/
│   └── banana_ml/
│       ├── __init__.py
│       ├── api.py
│       ├── api_schemas.py
│       │
│       ├── config/
│       │   └── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── validation.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   └── iot_features.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── predictor.py
│       │
│       └── utils/
│           └── __init__.py
│
├── tests/
│   ├── test_iot_validation.py
│   ├── test_iot_prediction.py
│   ├── test_predictor.py
│   └── test_api.py
│
├── scripts/
│   ├── train_with_mlflow.py
│   └── test_mlflow_model.py
│
├── models/
│   ├── development/
│   │   └── banana_quality_histgradientboosting.joblib
│   └── production/
│
├── configs/
├── reports/
│   ├── figures/
│   └── metrics/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pyproject.toml
└── uv.lock
```

---

# 7. 📡 Données IoT

Le schéma des mesures ESP32 est :

```text
id
device_id
temperature
humidity
mq2_raw
mq2_voltage
fc28_raw
liquid_detected
motion_detected
ldr_raw
light_label
detected_at
```

Exemple :

```json
{
  "id": 101,
  "device_id": "esp32-001",
  "temperature": 13.6,
  "humidity": 90.0,
  "mq2_raw": 800,
  "mq2_voltage": 0.64,
  "fc28_raw": 3000,
  "liquid_detected": 0,
  "motion_detected": 1,
  "ldr_raw": 2100,
  "light_label": "bright",
  "detected_at": "2026-10-06T02:30:00"
}
```

## Signification des capteurs

| Variable | Signification |
|---|---|
| `temperature` | température mesurée |
| `humidity` | humidité relative mesurée |
| `mq2_raw` | sortie analogique brute du MQ2 |
| `mq2_voltage` | tension calculée à partir de `mq2_raw` |
| `fc28_raw` | sortie analogique brute du FC28 |
| `liquid_detected` | indicateur de détection de liquide/humidité |
| `motion_detected` | indicateur de mouvement |
| `ldr_raw` | sortie analogique brute du LDR |
| `light_label` | interprétation de la luminosité |
| `detected_at` | timestamp de la mesure |

### ⚠️ Important concernant le MQ2

Le projet **ne considère pas `mq2_raw` comme une mesure directe d'éthylène**.

Le MQ2 est un capteur de gaz dont la réponse dépend de plusieurs facteurs et qui nécessite une calibration adaptée pour obtenir une concentration exploitable.

Dans le dataset actuel, `mq2_raw` est utilisé comme **signal capteur** associé à la qualité dans les données synthétiques.

Une calibration expérimentale sera nécessaire avant toute interprétation quantitative.

---

# 8. 🤖 Variables utilisées par le modèle

Le modèle IoT actuel utilise exactement les sept features suivantes :

```python
FEATURES = [
    "temperature",
    "humidity",
    "mq2_raw",
    "fc28_raw",
    "liquid_detected",
    "motion_detected",
    "ldr_raw",
]
```

La cible est :

```python
TARGET = "quality"
```

## Variables reçues mais non utilisées directement par le modèle

Certaines données font partie du schéma IoT mais ne sont actuellement pas envoyées au modèle :

```text
id
device_id
mq2_voltage
light_label
detected_at
```

`mq2_voltage` est dérivé de `mq2_raw`.

`light_label` est une représentation catégorielle de `ldr_raw`.

Ces variables pourront être utilisées dans de futures versions du pipeline.

---

# 9. 📊 Génération du dataset

Le dataset IoT utilisé pour le développement est :

```text
data/raw/banana_iot_model_ready.csv
```

Il contient :

```text
4500 lignes
14 colonnes
```

Répartition :

```text
bon       1500
moyen     1500
mauvais   1500
```

Les données ont été générées artificiellement afin de reproduire des profils cohérents avec les scénarios de stockage définis pour le projet.

Exemples de tendances utilisées :

```text
BON
→ température proche de 13–14 °C
→ humidité élevée
→ réponse MQ2 relativement faible
→ FC28 relativement élevé

MOYEN
→ conditions intermédiaires

MAUVAIS
→ humidité plus faible
→ réponse MQ2 plus élevée
→ FC28 plus faible
→ conditions thermiques plus défavorables
```

Ces relations sont des **hypothèses de génération du dataset**, et non des résultats expérimentaux établissant des relations biologiques réelles.

---

# 10. 📈 Analyse exploratoire

Les analyses sont réalisées dans :

```text
notebooks/01_data_exploration.ipynb
notebooks/02_iot_data_analysis.ipynb
```

Les analyses comprennent :

- statistiques descriptives ;
- distribution des classes ;
- distributions des capteurs ;
- corrélations ;
- comparaison des classes ;
- analyse des variables temporelles ;
- détection de valeurs aberrantes.

## Corrélations importantes

Certaines corrélations sont attendues car des variables sont dérivées les unes des autres.

Par exemple :

```text
mq2_raw ↔ mq2_voltage
```

est presque parfaitement corrélé puisque :

```text
mq2_voltage = mq2_raw / 4095 × 3.3
```

De même :

```text
ldr_raw ↔ light_bright
```

est fortement corrélé puisque `light_bright` est dérivé de `ldr_raw`.

Ces variables doivent donc être considérées avec attention lors de futures expérimentations pour éviter de compter plusieurs fois une information similaire.

---

# 11. 🤖 Modélisation

Deux modèles ont été évalués principalement :

- Random Forest ;
- HistGradientBoostingClassifier.

## Random Forest

Le modèle a été évalué avec les sept features IoT.

Features :

```python
[
    "temperature",
    "humidity",
    "mq2_raw",
    "fc28_raw",
    "liquid_detected",
    "motion_detected",
    "ldr_raw",
]
```

## HistGradientBoosting

Le modèle finalement enregistré dans MLflow est :

```text
HistGradientBoostingClassifier
```

avec :

```python
random_state=42
```

La séparation train/test utilise :

```python
test_size=0.2
random_state=42
stratify=y
```

---

# 12. 📊 Résultats expérimentaux

## Random Forest

Validation croisée 5-fold :

```text
Mean accuracy : 0.9878
Std            : 0.0036
```

Scores :

```text
0.9867
0.9878
0.9889
0.9822
0.9933
```

## HistGradientBoosting

Validation croisée 5-fold :

```text
Mean accuracy : 0.9896
Std            : 0.0019
```

Sur le test hold-out :

```text
Accuracy : 0.9878
```

Classification report :

```text
              precision    recall    f1-score

bon              0.99       0.99       0.99
mauvais          0.99       0.99       0.99
moyen            0.98       0.98       0.98
```

### ⚠️ Interprétation

Ces résultats sont **uniquement obtenus sur le dataset synthétique actuel**.

Ils démontrent que le pipeline de classification fonctionne correctement sur les distributions générées, mais ils ne permettent pas encore d'affirmer que le modèle possède une précision équivalente sur des mesures ESP32 réelles.

---

# 13. 🔬 MLflow

MLflow est utilisé pour suivre les expériences et gérer les versions du modèle.

Version actuelle :

```text
MLflow 3.16.1
```

Expérience :

```text
banana-quality-classification
```

Modèle enregistré :

```text
banana_quality_model
```

Version actuellement enregistrée :

```text
1
```

## Entraînement avec MLflow

```powershell
uv run python scripts/train_with_mlflow.py
```

Le script :

1. charge le dataset IoT ;
2. sélectionne les features ;
3. sépare les données ;
4. entraîne `HistGradientBoostingClassifier` ;
5. calcule l'accuracy ;
6. enregistre les paramètres ;
7. enregistre les métriques ;
8. enregistre le modèle dans le Model Registry.

Le modèle est enregistré avec :

```python
serialization_format="cloudpickle"
```

car la sérialisation initialement testée avec `skops` n'était pas compatible avec le modèle utilisé.

---

## Démarrer MLflow sous Windows

La configuration actuellement utilisée est :

```powershell
uv run mlflow server `
  --host 0.0.0.0 `
  --port 5000 `
  --workers 1 `
  --allowed-hosts "host.docker.internal:5000,localhost:5000,127.0.0.1:5000"
```

L'interface est disponible sur :

```text
http://127.0.0.1:5000
```

### Pourquoi `--workers 1` ?

Sur l'environnement Windows utilisé pour ce projet, le lancement multiprocess de MLflow a provoqué une erreur :

```text
WinError 10022
```

Le serveur est donc actuellement lancé avec un seul worker pour stabiliser l'environnement de développement.

---

# 14. 🚀 API FastAPI

L'API se trouve dans :

```text
src/banana_ml/api.py
```

Elle expose actuellement deux endpoints.

## Health check

```http
GET /health
```

Réponse :

```json
{
  "status": "ok",
  "service": "banana-quality-ml-api"
}
```

## Prediction

```http
POST /predict
```

L'API reçoit les données IoT et appelle :

```text
predict_quality()
```

qui :

1. valide la mesure ;
2. sélectionne les features ;
3. charge le modèle depuis MLflow Model Registry ;
4. effectue la prédiction ;
5. retourne la classe.

---

## Exemple de requête

```json
{
  "id": 101,
  "device_id": "esp32-001",
  "temperature": 13.6,
  "humidity": 90.0,
  "mq2_raw": 800,
  "mq2_voltage": 0.64,
  "fc28_raw": 3000,
  "liquid_detected": 0,
  "motion_detected": 1,
  "ldr_raw": 2100,
  "light_label": "bright",
  "detected_at": "2026-10-06T02:30:00"
}
```

Réponse attendue :

```json
{
  "device_id": "esp32-001",
  "quality": "bon",
  "model_version": "1.0.0"
}
```

> `model_version` est actuellement une valeur exposée par l'API et ne constitue pas encore une lecture dynamique de la version MLflow. La version réelle du modèle enregistré dans MLflow est actuellement `1`.

---

# 15. 🛡️ Validation des données

La validation est implémentée dans :

```text
src/banana_ml/data/validation.py
```

## Champs obligatoires

Pour une prédiction :

```text
temperature
humidity
mq2_raw
fc28_raw
liquid_detected
motion_detected
ldr_raw
```

## Contraintes analogiques

Les entrées analogiques ESP32 sont actuellement validées sur une plage ADC de 12 bits :

```text
0 ≤ mq2_raw ≤ 4095
0 ≤ fc28_raw ≤ 4095
0 ≤ ldr_raw ≤ 4095
```

## Variables binaires

```text
liquid_detected ∈ {0,1}
motion_detected ∈ {0,1}
```

Les données manquantes et les valeurs invalides sont rejetées.

---

# 16. 📡 Intégration ESP32

L'ESP32 constitue la source de données réelle du système.

Le format attendu correspond au schéma suivant :

```text
id
device_id
temperature
humidity
mq2_raw
mq2_voltage
fc28_raw
liquid_detected
motion_detected
ldr_raw
light_label
detected_at
```

## Exemple

```json
{
  "id": 101,
  "device_id": "esp32-001",
  "temperature": 13.6,
  "humidity": 90.0,
  "mq2_raw": 800,
  "mq2_voltage": 0.64,
  "fc28_raw": 3000,
  "liquid_detected": 0,
  "motion_detected": 1,
  "ldr_raw": 2100,
  "light_label": "bright",
  "detected_at": "2026-10-06T02:30:00"
}
```

## Données ESP32 actuellement disponibles

Les premières mesures de test comportent notamment :

```text
temperature = 0
humidity = 0
fc28_raw = 4095
ldr_raw = 4095
```

Ces valeurs sont considérées comme des données de test et/ou des indicateurs possibles de problème de mesure.

Elles ne doivent pas être ajoutées au dataset d'entraînement sans validation des capteurs.

---

# 17. 🧪 Tests

Le projet utilise `pytest`.

Exécuter tous les tests :

```powershell
uv run pytest -v
```

Tests de validation :

```powershell
uv run pytest tests/test_iot_validation.py -v
```

Tests de prédiction :

```powershell
uv run pytest tests/test_iot_prediction.py -v
```

Tests du predictor :

```powershell
uv run pytest tests/test_predictor.py -v
```

Tests API :

```powershell
uv run pytest tests/test_api.py -v
```

L'état actuel du projet comporte **10 tests validés** sur l'ensemble de la suite.

---

# 18. 🐳 Docker

L'API est conteneurisée avec Docker.

## Dockerfile

L'image utilise :

```text
python:3.12-slim
```

Le projet est installé à l'intérieur du conteneur avec `uv`.

Image actuelle :

```text
banana-quality-ml-api:1.0.0
```

## Build

```powershell
docker build -t banana-quality-ml-api:1.0.0 .
```

## Run

```powershell
docker run --name banana-quality-ml-api -p 8000:8000 banana-quality-ml-api:1.0.0
```

## Health check

```powershell
curl http://localhost:8000/health
```

---

## Docker et MLflow

L'API Docker utilise actuellement :

```text
http://host.docker.internal:5000
```

comme URI MLflow.

Cela permet au conteneur de communiquer avec le serveur MLflow exécuté sur la machine hôte Windows.

Le serveur MLflow doit donc être démarré avant d'utiliser `/predict` depuis le conteneur.

Architecture :

```text
┌──────────────────────────┐
│       Windows Host       │
│                          │
│  MLflow :5000            │
│  SQLite mlflow.db        │
│                          │
└────────────┬─────────────┘
             │
             │ host.docker.internal
             │
┌────────────▼─────────────┐
│      Docker Container    │
│                          │
│ FastAPI :8000            │
│ ML Predictor             │
│                          │
└──────────────────────────┘
```

> Cette architecture est adaptée au développement local. Pour une architecture de production, le stockage MLflow et les artifacts devront être externalisés et rendus accessibles de manière robuste au service ML.

---

# 19. 💻 Installation

## Prérequis

- Python 3.12
- uv
- Docker Desktop *(optionnel pour Docker)*
- Git

Vérifier Python :

```powershell
python --version
```

Vérifier uv :

```powershell
uv --version
```

Vérifier Docker :

```powershell
docker --version
```

## Installation

```powershell
git clone <repository-url>
cd model_ml_iot
```

Installer les dépendances :

```powershell
uv sync
```

---

# 20. ▶️ Exécution

## Option A — Exécution locale

### Terminal 1 : MLflow

```powershell
uv run mlflow server `
  --host 0.0.0.0 `
  --port 5000 `
  --workers 1 `
  --allowed-hosts "host.docker.internal:5000,localhost:5000,127.0.0.1:5000"
```

### Terminal 2 : API

```powershell
uv run uvicorn banana_ml.api:app --host 0.0.0.0 --port 8000
```

### Vérification

```powershell
curl http://localhost:8000/health
```

### Documentation

Ouvrir :

```text
http://localhost:8000/docs
```

---

## Option B — API dans Docker

### Terminal 1 : MLflow

```powershell
uv run mlflow server `
  --host 0.0.0.0 `
  --port 5000 `
  --workers 1 `
  --allowed-hosts "host.docker.internal:5000,localhost:5000,127.0.0.1:5000"
```

### Terminal 2 : Docker

```powershell
docker run --name banana-quality-ml-api -p 8000:8000 banana-quality-ml-api:1.0.0
```

### Vérifier l'API

```powershell
curl http://localhost:8000/health
```

---

# 21. ⚠️ Limites actuelles

## 21.1 Dataset synthétique

La principale limitation est l'utilisation de données synthétiques.

Les données actuelles ont été générées pour reproduire des profils cohérents avec les scénarios définis pour le projet.

Par conséquent :

> Une accuracy de 98–99 % sur ce dataset ne signifie pas que le modèle atteindra cette performance sur des données réelles.

Une validation expérimentale est nécessaire.

---

## 21.2 Nombre limité de données réelles

Les premières données ESP32 disponibles sont actuellement insuffisantes pour entraîner ou valider sérieusement le modèle.

Il faut collecter davantage de mesures :

```text
plusieurs jours
plusieurs conditions de stockage
plusieurs lots de bananes
plusieurs états de maturation
```

---

## 21.3 Calibration des capteurs

Les capteurs doivent être calibrés avant toute interprétation quantitative.

En particulier, le signal :

```text
mq2_raw
```

ne doit pas être présenté comme une concentration d'éthylène.

---

## 21.4 Classification instantanée

Le modèle actuel réalise une classification à partir d'une mesure.

Il ne prédit pas encore :

```text
qualité dans 24 h
qualité dans 48 h
qualité dans 72 h
```

Une telle fonctionnalité nécessiterait des données temporelles.

---

# 22. ⚙️ Évolution vers un système MLOps

L'architecture cible pourra évoluer vers :

```text
                    ESP32
                      │
                      ▼
                MQTT / HTTP
                      │
                      ▼
              IoT Data Ingestion
                      │
                      ▼
              Data Validation
                      │
                      ▼
              Feature Pipeline
                      │
                      ▼
                ML Prediction
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Quality      Risk       Anomaly
          │           │           │
          └───────────┼───────────┘
                      ▼
                  Database
                      │
                      ▼
                  Dashboard
```

## MLOps cible

```text
Git
 │
 ▼
CI/CD
 │
 ▼
Data Validation
 │
 ▼
Training
 │
 ▼
Evaluation
 │
 ▼
MLflow
 │
 ▼
Model Registry
 │
 ▼
Deployment
 │
 ▼
Monitoring
```

Composants envisagés :

- MLflow ;
- DVC ;
- Evidently ;
- Prometheus ;
- Grafana ;
- GitHub Actions ;
- PostgreSQL ;
- MQTT.

---

# 23. 🚀 Perspectives

## 23.1 Collecte de données réelles

Priorité principale :

```text
ESP32
  ↓
Collecte réelle
  ↓
Stockage
  ↓
Labelisation
  ↓
Training
  ↓
Validation
```

---

## 23.2 Détection d'anomalies

Détecter automatiquement des comportements inhabituels :

```text
Température anormale
Humidité anormale
Signal MQ2 inhabituel
FC28 inhabituel
Valeurs capteur incohérentes
```

---

## 23.3 Prédiction du risque

Créer un second modèle permettant d'estimer :

```text
LOW
MEDIUM
HIGH
```

pour le risque de dégradation.

---

## 23.4 Séries temporelles

Exploiter les séquences :

```text
t0 → t1 → t2 → t3 → ... → tn
```

pour étudier l'évolution de la qualité.

---

## 23.5 Explainable AI

Ajouter une explication aux prédictions :

```text
Prediction : MAUVAIS

Facteurs principaux :
- humidité faible
- réponse MQ2 élevée
- FC28 faible
- température défavorable
```

Ces explications devront être basées sur des méthodes d'explicabilité adaptées et non simplement sur une interprétation causale des feature importances.

---

## 23.6 Dashboard temps réel

À terme :

```text
┌──────────────────────────────────────┐
│       BANANA QUALITY MONITOR         │
├──────────────────────────────────────┤
│ Temperature       13.6 °C            │
│ Humidity          90.0 %             │
│ MQ2                800               │
│ FC28              3000               │
│ Motion              ON               │
│ Light             Bright             │
├──────────────────────────────────────┤
│ QUALITY             🟢 BON           │
└──────────────────────────────────────┘
```

---

# 24. 👨‍💻 Auteur

**Babacar Faye**

Master Intelligence Artificielle & Digital Computing  
Faculté des Sciences et Techniques — Université Sultan Moulay Slimane  
Béni Mellal, Maroc

Domaines d'intérêt :

```text
Artificial Intelligence
Machine Learning
Data Science
AI Engineering
Generative AI
AI Agents
Backend Engineering
Cloud & DevOps
MLOps
IoT
```

---

# 📄 Licence

Projet académique et expérimental.

La licence open-source pourra être définie lors de la publication du dépôt.