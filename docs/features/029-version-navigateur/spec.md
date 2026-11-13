# 029 — Version navigateur (« lite »)

## Objectif

Une version de l'app qui tourne **entièrement dans le navigateur**, sans serveur ni installation, hébergeable sur un site statique (GitHub Pages). Les données ne quittent jamais la page.

## Principe

Le **vrai backend** (FastAPI + le moteur `cyfouri-data-explorer`) tourne dans un Web Worker grâce à **Pyodide** (Python compilé en WebAssembly). Aucun calcul n'est réécrit : mêmes résultats que l'app et la bibliothèque.

| Pièce | Rôle |
|---|---|
| `lite/boot.py` | Crée l'app FastAPI dans Pyodide et répond aux requêtes par appel ASGI direct. Pas de threads en WebAssembly : routes synchrones, tâches de fond (`ThreadPoolExecutor`) et fichiers (`anyio.to_thread`) exécutés sur place. |
| `lite/bundle.py` | Emballe le moteur, `backend/app` et les exemples dans `python.zip`, et télécharge les 5 roues pures Python absentes de Pyodide (openpyxl, et-xmlfile, pydantic-settings, python-dotenv, holidays depuis 028 A5) : aucune visite n'appelle PyPI. |
| `frontend/src/lite/worker.ts` | Charge Pyodide (CDN jsDelivr, version figée) et les bibliothèques, décompresse `python.zip`, monte l'espace de travail en IndexedDB (persistant entre visites), sert les requêtes une par une. |
| `frontend/src/lite/bridge.ts` | Redirige `fetch("/api/…")` et les liens de téléchargement vers le worker. |
| `frontend/src/lite/EngineGate.tsx` | Écran de chargement avec progression. |
| `frontend/src/lite/flag.ts` | `LITE`, vrai seulement dans `npm run build:lite` (`vite build --mode lite`, `dist-lite/`, base relative, routes en `#`). |

## Ce qui est verrouillé (🔒)

| Élément | Raison |
|---|---|
| Machine learning | Entraînement, optimisation et explications trop lourds sur un seul cœur. |
| Données synthétiques | Modèles profonds (PyTorch) indisponibles ; le reste trop lent. |
| Assistant IA | Retiré de cette version. |
| « Browse files and folders » | Le navigateur ne lit pas les dossiers du disque : sélecteur ou glisser-déposer seulement. |
| ARIMA automatique, SARIMA | Trop lents dans le navigateur (la recherche d'ordres prend des minutes). |

Les modules verrouillés restent dans la barre latérale avec un cadenas ; leur page explique pourquoi et donne la commande pour installer l'app complète.

## Mesuré (2026-10-29, Pyodide 314.0.7)

- Démarrage : ~1 à 2 min **à chaque ouverture**, surtout pour compiler scipy, scikit-learn, statsmodels, DuckDB et pyarrow. Le téléchargement, lui, reste en cache.
- Ensuite les pages répondent en moins d'une seconde sur l'exemple de 600 lignes ; rapport HTML en 13 s.
- Vérifié : import (exemple et fichier de l'utilisateur), Overview, Columns, Variable, Compare, Correlations, Data et export CSV, Time series, KPI, rapport, persistance après rechargement.
- Limites : mémoire du WebAssembly (~2–4 Go), donc fichiers de taille moyenne seulement.

## Mise en ligne

`.github/workflows/lite.yml` construit et publie sur GitHub Pages à chaque push sur `master` (une fois activé : Settings › Pages › Source : GitHub Actions).
