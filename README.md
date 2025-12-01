# cv-intelligent-nlp-graph-learning
Analyse intelligente des offres d’emploi Data via NLP &amp; Graph Learning. Construction d’un graphe Postes–Compétences–Entreprises, moteur Q&amp;A Graph-RAG et adaptation automatique de CV basée sur des modèles Transformers.

# Intelligent CV & Job-Market Graph Analysis
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10+-blue)](#) [![Repo size](https://img.shields.io/github/repo-size/sarra2003/cv-intelligent-nlp-graph-learning)](#)

> Analyse avancée des offres d’emploi Data avec NLP, Graph Learning, moteur sémantique et module d’adaptation automatique de CV.

---

## 📌 Résumé
Ce projet exploite des offres d’emploi pour :
- extraire compétences / technologies / entreprises (NLP),
- classifier les postes (Transformers),
- construire un graphe Postes–Compétences–Entreprises,
- appliquer GNN pour recommandations et détection de communautés,
- proposer un module Graph-RAG (mini Q&A) et un module d’adaptation automatique de CV (suggestions & reformulations).

---

## 🎯 Objectifs
- Analyse des tendances du marché Data
- Recommandation de compétences pour un poste
- Personnalisation sémantique de CV
- Recherche sémantique / Q&A sur le graphe

---

## 🧠 Architecture (schéma)
TEXT DATA (job offers)
│
▼
NLP Pipeline (clean → NER → embeddings → classification)
│ │
▼ ▼
jobs table skills list
\ /
\ /
▼ ▼
Knowledge Graph (Jobs–Skills–Companies)
│
GNN / community detection
│
Recommendations / Graph-RAG Q&A
│
CV Adaptation Module (suggestions, rewrite)

---

## 🔧 Structure du repo (recommandée)
cv-intelligent-nlp-graph-learning/
├── src/
│ ├── data_processing/
│ ├── nlp/
│ ├── graph/
│ ├── gnn/
│ ├── rag/
│ ├── cv_adaptation/
│ └── api/
├── requirements.txt
├── .gitignore
├── README.md
└── .env.example

yaml
Copier le code

---

## ⚙️ Installation rapide

> Utilise Python 3.10+

```bash
git clone https://github.com/sarra2003/cv-intelligent-nlp-graph-learning.git
cd cv-intelligent-nlp-graph-learning
python -m venv .venv
# Windows
.venv\Scripts\activate
# mac/linux
# source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
 Lien dataset : https://huggingface.co/datasets/lukebarousse/data_jobs

