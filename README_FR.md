# TCE-Driven Control for Timely V2X Safety Communications

[English](README.md) | Français

## 1. Présentation du projet

Ce dépôt contient la version candidate destinée à la diffusion d'un simulateur
analytique et stochastique au niveau système. Il sert à étudier la fiabilité et
la latence des messages de sécurité V2X lorsque les conditions de propagation
se dégradent ou que le réseau est chargé. Trois scénarios sont évalués (`Ref`,
`UrbMask` et `Tunnel`), avec ou sans congestion (`NoCong` ou `Cong`), ainsi que
cinq stratégies de retransmission.

La métrique centrale est la Timely Communication Effectiveness (TCE). Elle
distingue les réceptions strictement dans les délais de celles qui arrivent en
retard mais conservent encore une utilité partielle. Ce code accompagne les
travaux de recherche *TCE-Driven Control for Timely V2X Safety Communications*.

Il ne s'agit pas d'une pile protocolaire V2X. Les limites du modèle sont
détaillées dans la section [Périmètre du modèle et limites](#13-périmètre-du-modèle-et-limites).

## 2. Qu'est-ce que la TCE ?

Pour un paquet physiquement reçu avec un délai `d`, une échéance `D`, un
intervalle de grâce `G`, un coefficient de décroissance `beta` et un exposant
`gamma`, l'utilité du paquet est définie par :

```text
u(d) = 1                                      if d <= D
       exp[-beta ((d-D)/G)^gamma]             if D < d <= D+G
       0                                      if d > D+G
```

L'utilité d'un paquet physiquement perdu est nulle. La TCE correspond à
l'utilité moyenne des paquets. L'implémentation recalcule le respect de
l'échéance à partir de la réception physique et de l'échéance active pour
l'analyse. À la précision numérique près, elle vérifie donc :

```text
Timely reception rate <= TCE <= PHY reception rate
```

Un PDR seul traite de la même manière tous les paquets physiquement reçus. La
TCE conserve la distinction entre une réception dans les délais, une réception
tardive mais encore utile, et une réception inutile ou perdue. Les frontières
exactes sont décrites dans [Methodology](docs/METHODOLOGY.md).

## 3. Structure du dépôt

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── paper_s15.json
├── docs/
├── examples/
│   ├── run_reviewer_smoke.py
│   └── run_paper_configuration.py
├── models/
│   ├── MODEL_MANIFEST.csv
│   └── six S15 empirical MDP-lite JSON models
├── matlab/
│   └── plot_s15_paper_figures.m
└── py/
    ├── run_pipeline_C.py
    ├── sim_v2x_C.py
    ├── analyze_metrics_C.py
    ├── analyze_policy_C.py
    ├── analyze_tce_C.py
    ├── compare_policies_C.py
    ├── study_readiness_C.py
    ├── build_mdp_lite_table_C.py
    └── modules/
        ├── retx_policy.py
        ├── tce_metric.py
        └── propagation, congestion, traffic, and scenario modules
```

Principaux points d'entrée :

| Fichier | Rôle |
|---|---|
| `run_pipeline_C.py` | Chaîne unifiée : génération → simulation → analyse → contrôle de pertinence → tracés. |
| `sim_v2x_C.py` | Simulation stochastique au niveau paquet et génération du journal des décisions. |
| `modules/retx_policy.py` | Décisions des stratégies NoRet, Classical, Nomikos, UDRC et MDP-lite. |
| `modules/tce_metric.py` | Configuration de référence de la TCE, utilité par paquet et agrégation. |
| `analyze_metrics_C.py` | Métriques conventionnelles sur les paquets et classes de distance. |
| `analyze_policy_C.py` | Synthèses par graine des stratégies et de la TCE, avec diagnostics des décisions. |
| `analyze_tce_C.py` | Synthèses TCE, tableaux par distance ou bande et utilités compactes par paquet. |
| `compare_policies_C.py` | Regroupe les synthèses des stratégies dans des tableaux comparatifs. |
| `study_readiness_C.py` | Vérifie si une exécution D/G met en évidence une zone cachée pertinente pour l'étude ; ce contrôle porte sur l'intérêt scientifique du cas, pas sur la correction logicielle. |
| `build_mdp_lite_table_C.py` | Construit hors ligne une table empirique de transition MDP-lite. |
| `matlab/plot_s15_paper_figures.m` | Script final produisant les sept figures S15 + MDP. |

Les fichiers générés sont placés sous `workspace/`, qui est exclu de Git.

## 4. Installation

La version candidate a été validée avec Python 3.11.9 et les versions exactes
des paquets indiquées dans `requirements.txt`.

Windows PowerShell :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Le dossier `.venv` ne doit pas être ajouté au dépôt. MATLAB est facultatif et
n'est utilisé que par le script conservé pour les figures. La version de MATLAB
n'avait pas été formellement figée lors des expériences d'origine.

## 5. Démarrage rapide

Depuis la racine du dépôt, exécutez le test rapide validé pour les reviewers :

```powershell
python examples/run_reviewer_smoke.py
```

Ce programme appelle la véritable chaîne de simulation pour une petite
expérience `Ref`/Classical : une graine, une trajectoire de deux secondes et le
mode sûr `tx_mode=mix`. Il vérifie ensuite que :

- la sortie contient des paquets ;
- le schéma requis des paquets est présent ;
- les synthèses de stratégie et de TCE existent ;
- l'inégalité `Timely <= TCE <= PHY` est respectée ;
- le manifeste conserve la chaîne, la TCE, les étapes et la provenance des entrées.

En cas de succès, l'exécution se termine par `REVIEWER SMOKE PASS` et affiche un
identifiant d'exécution. En cas d'échec, elle renvoie un code non nul. Pour
ajouter la validation stricte de MDP-lite :

```powershell
python examples/run_reviewer_smoke.py --with-mdp
```

Ce second test charge le modèle NoCong/Ref fourni et exige, dans le journal des
décisions, un taux de correspondance au modèle égal à 1, aucun échec de
recherche et aucun fallback.

## 6. Stratégies de retransmission

| Stratégie | Signification dans l'implémentation |
|---|---|
| NoRet | Ne poursuit jamais après l'échec de la première tentative. |
| Classical | Retransmet tant qu'un budget reste disponible. |
| Nomikos | Référence conceptuelle inspirée de la littérature et fondée sur une échéance stricte ; la retransmission n'est autorisée que si la prochaine arrivée prédite reste dans les délais. Il ne s'agit pas d'une reproduction complète d'un protocole publié. |
| UDRC | Retransmet lorsque le gain attendu d'utilité TCE sur la chaîne restante dépasse le coût normalisé de retransmission, pondéré par lambda. |
| MDP-lite | Référence simplifiée guidée par un modèle, utilisant une table empirique de transition et une valeur récursive à horizon court. Ce n'est pas un ordonnanceur MDP complet. |

Le chemin d'exécution recommandé pour les reviewers et les expériences formelles
échoue si MDP-lite est sélectionné sans modèle valide. Le fallback historique
vers la chaîne n'est disponible qu'avec l'option avancée explicite
`--mdp_allow_chain_fallback`. Cette option ne doit pas être utilisée pour
revendiquer des résultats effectivement guidés par le modèle.

## 7. Configuration formelle de l'étude

Les journaux de commandes S15 conservés établissent la configuration de
recherche suivante :

| Paramètre | Valeur formelle |
|---|---|
| Scénarios | Ref, UrbMask, Tunnel |
| Charges | NoCong et Cong |
| Durée / pas de temps / graines | 60 s / 0.1 s / graines 1–15 |
| Fréquence des messages / distance maximale | 10 Hz / 200 m |
| TCE | D=20 ms, G=50 ms, beta=ln(20), gamma=2 |
| Sélection TX | mix, k=6, cross k=2 |
| UDRC | lambda=0.03, mode de coût `delay_cbr` |
| MDP-lite | ret=2, seuil=0, échelle de coût=0.85, minimum d'échantillons=3, facteur d'actualisation=1 |

Ces valeurs décrivent les expériences de recherche ; elles ne correspondent
pas nécessairement aux valeurs par défaut de l'interface en ligne de commande.
Elles sont enregistrées dans `configs/paper_s15.json`. Pour examiner une
commande formelle protégée sans lancer le calcul complet :

```powershell
python examples/run_paper_configuration.py --scenario UrbMask --load Cong --policy udrc
```

Par défaut, ce programme affiche seulement la commande. L'option `--execute`
est nécessaire pour démarrer explicitement l'expérience à 15 graines.

## 8. Modifier la configuration expérimentale

| Objectif | Option | Signification | Exemple sûr |
|---|---|---|---|
| Scénario | `--scenarios` | Liste Ref/UrbMask/Tunnel | `--scenarios Ref` |
| Charge | `--enable_congestion` | Omettre pour NoCong ; ajouter pour Cong | `--enable_congestion` |
| Stratégie | `--retx_policy` | L'une des cinq stratégies | `--retx_policy udrc` |
| Budget de retransmission | `--rets` | Tentatives supplémentaires : 0, 1 ou 2 | `--rets 2` |
| Graines | `--seed_start`, `--n_seeds` | Graines NumPy consécutives | `--seed_start 1 --n_seeds 3` |
| Durée | `--duration_s` | Durée de la trajectoire générée | `--duration_s 10` |
| Fréquence des messages | `--msg_rate_hz` | Messages de sécurité par seconde | `--msg_rate_hz 10` |
| Portée de communication | `--max_distance_m` | Filtre strict de sortie sur la distance TX–RX | `--max_distance_m 200` |
| Échéance TCE | `--tce_deadline_ms` | Limite d'une réception strictement dans les délais | `--tce_deadline_ms 20` |
| Grâce TCE | `--tce_grace_ms` | Intervalle d'utilité partielle | `--tce_grace_ms 50` |
| Compromis UDRC | `--udrc_lambda` | Poids du coût | `--udrc_lambda 0.03` |
| Définition du coût | `--policy_cost_mode` | Mode fondé sur délai/temps d'antenne/CBR/p_col | `--policy_cost_mode delay_cbr` |
| Modèle MDP | `--mdp_model_path` | Table empirique JSON | `--mdp_model_path models/...json` |
| Coût MDP | `--mdp_cost_scale` | Échelle appliquée en ligne au coût empirique | `--mdp_cost_scale 0.85` |

Le scénario, la charge, le nombre de graines, la durée et la portée de sortie
sont des paramètres expérimentaux courants. Modifier D/G/beta/gamma, le lambda
ou la définition du coût UDRC, l'échelle MDP, ou encore les équations de
propagation, de trafic ou de congestion, change la configuration scientifique.
Une telle exécution ne doit pas être présentée comme une reproduction de S15
sans nouveaux éléments de preuve.

## 9. Fichiers de sortie

Chaque exécution est enregistrée dans `workspace/results/runs/<run_id>/` :

- `raw/results_packets...csv` : une ligne par résultat message/TX/RX ;
- `raw/results_retx_decisions...csv` : décisions de stratégie et diagnostics pour chaque échec ;
- `tables/policy_summary...csv` : métriques Timely/PHY/TCE et stratégie, agrégées sur les graines ;
- `tables/policy_compare...csv` : synthèses comparatives des stratégies ;
- `tables/tce_summary...csv` : décomposition TCE globale alignée sur l'étude ;
- `tables/readiness...csv` : contrôle de la zone cachée et de la séparation ;
- `figures/` : figures Python facultatives ;
- `run_manifest.json` : provenance cumulative des étapes, configurations, entrées et modèles ;
- `run_commands.txt` : journal local d'exécution, exclu de Git.

Voir [Output schema](docs/OUTPUT_SCHEMA.md).

## 10. Reproduire des résultats de type S15

L'étude S15 complète couvre les combinaisons scénarios × charges × stratégies
avec 15 graines. Elle a été produite par des exécutions séparées pour chaque
stratégie, partageant les mêmes entrées de scénario, puis par une comparaison
des stratégies et des tracés MATLAB. Utilisez `configs/paper_s15.json` et le
programme formel en mode affichage pour construire des commandes explicites ;
n'utilisez pas `latest` pour une analyse formelle.

Les quelque 40 Go de sorties brutes historiques ne sont pas inclus. Ce dépôt
contient le code source et les six petits modèles empiriques MDP-lite nécessaires
pour régénérer de nouvelles exécutions. Une reproduction complète demande des
ressources de calcul et de stockage importantes et ne doit pas être confondue
avec le test rapide destiné aux reviewers.

## 11. Modèles MDP-lite

Six modèles JSON se trouvent dans `models/` : NoCong/Cong ×
Ref/UrbMask/Tunnel. Leurs sommes de contrôle et leur provenance sont indiquées
dans `models/MODEL_MANIFEST.csv`. Ils ont été construits hors ligne à partir des
données de décisions et de paquets Classical avec ret=2 et les graines 1–15.
Pour les six combinaisons, les exécutions formelles S15 ont enregistré un taux
de correspondance au modèle de 1.0, aucun échec de recherche et aucun fallback.

Ces modèles n'ont pas été évalués dans le cadre d'une étude de généralisation
avec des jeux d'entraînement et de test indépendants : les tables empiriques et
les évaluations formelles utilisent la même plage de graines. Le modèle choisi
doit correspondre à la fois au scénario et à la charge.

## 12. Notes de calcul

- La TCE demande O(1) en temps et O(1) en état auxiliaire par paquet.
- Nomikos demande O(1) par décision.
- UDRC demande O(R) par décision pour le budget restant R ; R<=2 dans S15.
- La recherche et la récursion de valeur MDP-lite en ligne sont prévues en O(R) pour un ensemble borné de clés de dictionnaire.
- La construction de la table MDP est effectuée hors ligne avec des opérations DataFrame de tri, fusion et regroupement ; une borne prudente est O(N log N) en temps et O(N) en mémoire pour N lignes de décision sources.

## 13. Périmètre du modèle et limites

Le simulateur repose sur des abstractions analytiques et stochastiques au niveau
système pour la propagation, la réception, le délai et la congestion. Il ne
constitue pas une pile protocolaire complète 802.11p, C-V2X ou NR-V2X. Il
n'implémente pas HARQ, SPS, un ordonnancement réaliste des ressources, ns-3,
OMNeT++, Veins ou SUMO.

La TCE actuelle ne modélise pas explicitement l'AoI, l'intervalle entre
réceptions, une utilité sensible à la burstiness, l'utilité des pertes
consécutives, l'historique de fraîcheur ou l'évolution de l'état applicatif. Une
source fixe à 10 Hz n'est pas équivalente à un modèle applicatif tenant compte
de la burstiness.

## 14. Reproductibilité

Les composantes aléatoires de la simulation, des trajectoires, des bâtiments et
des variations de lien utilisent des graines NumPy explicites. Le manifeste de
la version candidate conserve des références relatives vers les modèles et
leurs valeurs SHA-256. `requirements.txt` fixe l'environnement Python validé.
Les exécutions formelles doivent utiliser des identifiants, configurations,
modèles et graines explicites, plutôt qu'une sélection fondée sur la date de
modification ou sur `latest`. Voir [Reproducibility](docs/REPRODUCIBILITY.md).
