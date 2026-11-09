# Architecture des dossiers

Objectif : pouvoir **ajouter une page, un module, un visuel, un test statistique ou un rôle de colonne sans toucher au reste**. Les dossiers marqués *(futur)* n'existent pas encore : ils montrent où les choses iront.

## Arborescence

```
New App/
├── CLAUDE.md · PLAN.md · README.md (EN) · LICENSE (MIT) · .env.example · .gitignore
├── Dockerfile                    ← multi-étapes : build du front (Node) → image Python slim
├── docker-compose.yml            ← port 127.0.0.1:8000, volume workspace, ./data monté en /data (lecture seule)
├── docker-compose.gpu.yml        (futur) variante CUDA pour ML / synthétique
├── docs/                         ← voir CLAUDE.md pour l'index
├── samples/                      ← démos commitées : titanic, iris, penguins, california_housing, typing_traps, série temporelle
├── scripts/                      ← dev.ps1, make_big_sample.py (≈5 M lignes, non commité)…
│
├── packages/
│   └── quantilo/    ← LE MOTEUR, bibliothèque Python (027) : ni FastAPI, ni SQLite, ni serveur
│       ├── pyproject.toml        ← nom pip quantilo, extras [ml] et [deep], son propre uv.lock
│       ├── README.md · examples/quickstart.ipynb
│       ├── src/quantilo/
│       │   ├── __init__.py       ← API publique : open(), from_pandas(), Dataset, to_dict(), to_pandas(), erreurs
│       │   ├── dataset.py        ← Dataset : catalogue en mémoire + façade de toutes les analyses
│       │   ├── catalog.py        ← protocoles Catalog / DatasetRef, ColumnView, règles de correction des colonnes
│       │   ├── errors.py         ← ExplorerError (message, code, statut HTTP que l'app renverra), ColumnNotFound
│       │   ├── results.py        ← to_dict() / to_pandas()
│       │   ├── plots/            ← dx.plot() : les figures de l'app portées en Python (extra [plot]),
│       │   │                       une par fonction, comme frontend/src/**/figures.ts ; theme.py = tokens
│       │   ├── ingestion/        ← sources, détection CSV, lecture DuckDB, aperçu, import Parquet, navigation
│       │   ├── typing/           ← moteur de détection + knowledge_base/ (roles.toml, vocabulary.toml)
│       │   ├── profiling/ · variable/ · compare/ · correlations/ · charts/ · filters/ · rows/
│       │   ├── changes/          ← ce qui a changé entre deux versions (031) : photo, PSI / KS / χ², lignes
│       │   ├── recipes/          ← recettes (030) : schéma versionné, correspondance des colonnes,
│       │   │                       rejeu (protocole Target), export Python ; dx.Recipe
│       │   ├── report/           ← rapport HTML (build, svg, html) + generate.py (enchaîne les analyses)
│       │   ├── timeseries/       ← méthodes + service.py (page Séries temporelles) + facade.py (bibliothèque)
│       │   ├── kpi/              ← calcul, règles, suggestions + service.py (KpiEngine)
│       │   ├── experiments/      ← tests A/B (stats, analyze, plan) + service.py (ExperimentEngine) + facade.py
│       │   ├── ml/               ← préparation, catalogue, entraînement, explications, optimisation, erreurs (036), Auto (auto.py, 038)
│       │   │                       + engine.py (MlEngine, synchrone) + facade.py (bibliothèque)
│       │   ├── synthetic/        ← méthodes, rapport, confidentialité + engine.py (SyntheticEngine) + facade.py
│       │   ├── segments/         ← segments et lignes inhabituelles (032, 033) : préparation « distance » du ML, méthodes,
│       │   │                       portraits, carte, scores des lignes + engine.py (SegmentsEngine) + facade.py
│       │   └── gpu.py            ← détection du GPU
│       └── tests/                ← tests unitaires du moteur + test_dataset.py / test_facades.py
│
├── backend/                      ← L'APP : API web, état (SQLite), tâches de fond, assistant
│   ├── pyproject.toml            ← dépend de quantilo[ml] (source uv : chemin, éditable)
│   ├── app/
│   │   ├── main.py               ← crée l'app FastAPI, monte les routers, sert le front buildé
│   │   ├── config.py             ← réglages (.env) : workspace, seuils, limites
│   │   ├── core/                 ← transverse : db.py (SQLite + migrations/), storage.py, jobs.py,
│   │   │                           errors.py (AppError = ExplorerError du moteur), tools.py (contrat d'outil de l'agent)
│   │   ├── services/             ← ce qui garde un état : datasets/ (cycle de vie, jobs d'import),
│   │   │                           columns/ (corrections en SQLite ; satisfait le protocole Catalog),
│   │   │                           filters/saved.py, settings/, report/ (jobs du rapport),
│   │   │                           versions/ (photo de chaque import, version précédente gardée, 031)
│   │   ├── api/                  ← routers HTTP fins : appellent le moteur et services/, ne calculent rien
│   │   ├── schemas/              ← modèles Pydantic = contrats de l'API
│   │   └── modules/              ← modules optionnels, chacun autonome (router, service, agent_tools)
│   │       ├── registry.py       ← liste des modules actifs → routers montés
│   │       ├── timeseries/ · kpi/ ← routers sur les services du moteur (+ KPI enregistrés)
│   │       ├── experiments/      ← router + outils de l'agent sur ExperimentEngine (rien de stocké)
│   │       ├── ml/ · synthetic/  ← MlService(MlEngine), SyntheticService(SyntheticEngine) : jobs + runs enregistrés
│   │       ├── segments/         ← SegmentsService(SegmentsEngine) : jobs, runs, marques, nouveaux datasets
│   │       ├── recipes/          ← recettes (030) : rejeux en tâche de fond par les services des autres
│   │       │                       modules (lus sur app.state, jamais importés)
│   │       └── agent/            ← assistant IA (moteur pur, fournisseurs, outils)
│   └── tests/
│       ├── unit/                 ← migrations, contrat des outils, agent
│       └── api/                  ← routes HTTP, dont test_library_parity.py (app = bibliothèque)
│
└── frontend/
    ├── package.json · vite.config.ts · tsconfig.json
    ├── src/
    │   ├── app/                  ← point d'entrée, router, providers (Query, thème), layout (Shell, Sidebar, TopBar)
    │   ├── pages/                ← une route = un dossier
    │   │   └── home/ · columns/ · overview/ · variable/ · compare/ · correlations/ · data/ · settings/
    │   ├── features/             ← blocs métier réutilisés par plusieurs pages
    │   │   └── filters/ · type-badge/ · summary-strip/ · chart-card/ · variable-grid/ · command-palette/
    │   ├── charts/               ← couche Plotly unique
    │   │   ├── Plot.tsx          ← wrapper (chargement paresseux, redimensionnement, thème)
    │   │   ├── theme.ts          ← tokens CSS → layout Plotly
    │   │   ├── rules/            ← règles d'affichage (Top N, orientation, échantillonnage)
    │   │   ├── builders/         ← un visuel par fichier
    │   │   └── registry.ts       ← types × intention → visuels possibles
    │   ├── modules/              ← pendant front des modules backend
    │   │   ├── registry.ts       ← manifestes → routes + entrées de la barre latérale
    │   │   └── timeseries/ (futur) manifest.ts · pages/ · components/
    │   ├── ui/                   ← design system : Button, Select, Dialog, Tooltip, Tabs, Table… (aucune logique métier)
    │   ├── api/                  ← client typé généré depuis l'OpenAPI du backend + hooks par ressource
    │   ├── stores/               ← état UI (filtres actifs, préférences)
    │   ├── lib/                  ← formatage des nombres (selon la locale), i18n, utilitaires
    │   └── styles/               ← tokens.css, polices locales
    └── tests/e2e/
```

## Règles de dépendance

```
Frontend :  pages ─▶ features ─▶ charts ─▶ ui          modules ─▶ (features, charts, ui)
Backend  :  api ─▶ services ─▶ core                     modules ─▶ (services, core)
            tout ─▶ quantilo (le moteur, qui ne connaît rien de l'app)
```

- `ui/` ne connaît rien du métier. Le moteur ne connaît rien de HTTP ni de SQLite : un test le vérifie (`test_the_package_never_imports_the_app`), et la CI l'installe seul (job `library`).
- **Un calcul vit dans le moteur** (`packages/quantilo`). Il prend des entrées simples : une connexion DuckDB et une relation, ou des tableaux, plus des descriptions de colonnes. Il rend des dataclasses. Un service d'analyse du moteur demande les colonnes à un `Catalog` : `ColumnService` (SQLite) dans l'app, `Dataset` (en mémoire) dans la bibliothèque. Les deux appliquent les mêmes règles (`catalog.apply_change`).
- **L'app ajoute l'état** (`services/`, `modules/*/service.py`) : persistance, jobs, caches. Elle hérite des moteurs synchrones (`MlEngine`, `SyntheticEngine`, `KpiEngine`) au lieu de recopier leurs calculs.
- **Un module n'importe jamais un autre module.** Ce qui doit être partagé descend dans `features/` ou dans le moteur.
- Les pages n'appellent pas `fetch` directement : elles passent par `api/`.

## Recettes d'extension

| Je veux ajouter… | Je fais |
|---|---|
| Une analyse dans la bibliothèque | Le calcul dans le moteur, puis une méthode de `Dataset` (`dataset.py`) qui appelle le service avec `self._id` et `self.conditions`, un test dans `packages/quantilo/tests/`, une ligne dans le README du paquet. |
| Une page V1 | `frontend/src/pages/<nom>/` + une route dans `app/router.tsx` + `docs/pages/NN-nom.md` |
| Un module (ex. séries temporelles) | `backend/app/modules/<nom>/` (router, service) + l'enregistrer dans `modules/registry.py` avec son `setup(app)` (crée ses services sur `app.state`) ; `frontend/src/modules/<nom>/manifest.ts` : `scope: "global"` (`/modules/<nom>`) ou `scope: "dataset"` (`/d/$datasetId/<nom>`, `requires` = types de colonnes nécessaires, grisé sinon), `validateSearch` optionnel, + l'enregistrer dans `modules/registry.ts`. La barre latérale et les routes apparaissent seules. Le calcul va dans le moteur (ex. `quantilo/timeseries/`, utilisé par Comparer et le module), avec une méthode sur `Dataset` pour la bibliothèque. |
| Un panneau à côté des pages (ex. l'assistant, 026) | Un composant qui décide lui-même quand s'afficher + un `PanelManifest` (`id`, `component`) listé dans `panels` de `frontend/src/modules/registry.ts` ; le Shell l'affiche à droite du contenu, sur toutes les pages. |
| Un module isolé avec son propre moteur (ex. l'agent, 026) | `backend/app/modules/agent/` : `engine/` pur (ni FastAPI, ni SQLite, ni SDK : testable avec un fournisseur simulé ; boucle, budget, prompt, vérificateur d'ancrage `grounding.py`, `compaction.py`), `playbooks/` (méthodes en markdown, lues par l'outil `get_playbook`), `providers/` (SDK derrière l'interface `Provider`), `tools/` (chaque outil appelle les services V1 injectés via `AppServices`, jamais un autre module), `service.py` / `router.py` / `repository.py` autour. Le reste de l'app ne le connaît pas : une ligne dans `modules/registry.py`, une migration, un panneau. |
| Un outil de l'agent dans un module (026b) | `app/modules/<nom>/agent_tools.py` : des classes `Tool` (contrat partagé `app/core/tools.py` : entrée Pydantic, `kind` read / compute / write, sortie compacte + lien app, `wait_for` pour suivre une tâche de fond, `compact` pour alléger un résultat) et une fonction `agent_tools(app)`, déclarée dans `Module(agent_tools=…)` de `modules/registry.py`. Le module n'importe pas l'agent ; l'agent ramasse ces outils au démarrage. Un outil `write` est toujours confirmé par l'utilisateur. |
| Un visuel | Sa figure dans le frontend (`figures.ts` de la page ou du module) **et** sa jumelle dans `quantilo/plots/` (branchée dans `plots/__init__.py`, testée dans `test_plots.py`) ; `frontend/src/charts/builders/<visuel>.ts` (+ préparation dans `quantilo/charts/` si besoin), voir [VISUALS.md](VISUALS.md) |
| Un test statistique | Une fonction dans le moteur, à côté de sa famille (`quantilo/compare/stats.py`, `variable/numeric.py`, `timeseries/…`), testée contre l'implémentation de référence (SciPy, statsmodels) ; l'app et la bibliothèque la voient ensemble. |
| Un rôle de colonne | Une entrée TOML dans `quantilo/typing/knowledge_base/roles.toml` + un cas de test, voir [COLUMN_DETECTION.md](COLUMN_DETECTION.md) |
| Une table SQLite | Une migration numérotée dans `core/migrations/` + mise à jour de [BACKEND_SCHEMA.md](BACKEND_SCHEMA.md) |
| Modifier une fonctionnalité existante | Mettre à jour d'abord sa doc (`docs/pages/…`), ensuite le code et les tests |
