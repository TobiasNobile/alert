# ALERT : Anomaly Localization & Early Real-Time Detection

Alert est un logiciel de démonstration clé en main, libre d’accès, destiné aux usages académiques et industriels, permettant de détecter et de localiser en temps réel des anomalies dans un signal vidéo.

## Démonstration Vidéo

<iframe
  src="https://webtv.centralesupelec.fr/permalink/v126d5bdf432e5dhy5cy/iframe/?notabs#start=163&autopilot=no"
  width="800"
  height="450"
  allowfullscreen>
</iframe>

## Démarrage rapide

```shell
python main.py            # puis ouvrir http://127.0.0.1:8000
```

Le fit et le scoring s'excluent : les deux occupent le thread principal, seul
endroit où torch est sûr sur macOS.

## La page

Une page, deux moitiés.

**À gauche, on construit une banque** : un nom de tâche, un backbone, les
couches, le taux de coreset, et des images sans l'anomalie à détecter — un zip ou
un dossier. Un sous-dossier `anomaly/` est reconnu et gardé hors banque, il ne
sert qu'à calibrer un seuil. 20 % des images normales sont elles aussi réservées
hors banque, de quoi juger la banque sans lui redonner ses propres images — une
image de la banque est son propre plus proche voisin et score presque zéro. À la
fin du fit, le `.pkg` apparaît dans `coresets/` et dans le sélecteur de droite.

**À droite, on score la caméra** : on choisit une banque, une source, et ça
tourne. Se règlent sans rien arrêter :

| réglage | ce qu'il fait |
| --- | --- |
| **stride** | ne score qu'une frame sur *n* |
| **zoom** | recadrage centré ; 1 = pleine image |
| **vmax** | le pied de la couleur : rien n'est peint en dessous, la couleur monte avec ce qui dépasse |
| **opacité** | la vitesse à laquelle la couleur monte au-dessus de vmax |
| **lissage** | agrège les cartes sur une tranche de vidéo, en moyenne ou en max |

Le choix du device (`auto`, `cpu`, `cuda`, `mps`), les threads faiss et faiss sur
GPU se posent avant de lancer le scoring.

`vmax` est le réglage qui décide de ce qui s'affiche. Le score de PatchCore est
une distance, sans unité : son ordre de grandeur dépend de la couche, du backbone
et de la scène, et une valeur ne transfère pas d'une banque à l'autre. La page
pré-remplit le champ depuis une table par couche (`l2: 20`, `l3: 10`, `l4: 260`,
`l3-l4: 150–200`) qui suit les cases cochées à gauche, et affiche ce repère sous
le champ. C'est un point de départ, à remonter ou descendre en marche jusqu'à ce
que le décor cesse de se teinter.

Un fichier vidéo est lu **à sa vitesse réelle** : la boucle se cale sur l'horloge
de la source, attend si elle est en avance et saute des frames si elle est en
retard. Sans ça le clip défilerait au rythme du traitement — au ralenti à stride
bas, en accéléré à stride haut — et deux réglages de lissage ne seraient plus
comparables. Le prix est visible dans la page : à stride 1 sur un clip 60 fps,
seule une frame sur dix environ est scorée si l'inférence prend 170 ms.

## Les banques livrées

Le dépôt livre quatre banques dans `coresets/`, prêtes à scorer : la même tâche
« personne + couteau » sur quatre backbones — WideResNet50, ResNet50, ResNet34,
ResNet18 — à configuration identique par ailleurs (224 px, l3-l4, coreset 0,005,
20 000 images des splits `train2017` et `val2017`, seed 0) : de quoi voir en
direct ce que coûte un backbone plus léger. La page présélectionne WideResNet50,
le seul des quatre au-dessus de 0,6 d'AUROC. Toute autre banque déposée dans
`coresets/` est trouvée au démarrage suivant.

## Cadence live — coût d'une frame

Banque « personne + couteau » (COCO, layer3 + layer4, 20 000 images de fit,
224 px). Budgets : 33,3 ms = 30 FPS, 16,7 ms = 60 FPS. `scoring` exclut
l'encodage JPEG.

CPU (Apple M-series, torch 4 threads / faiss 1) :

| backbone | coreset | vecteurs en banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | p0.005 | 19 600 | 2,5 ms | 54,5 ms | 16,4 ms | 2,7 ms | 1,0 ms | 76,2 ms | 12,9 | 0,6321 |
| wideresnet50 | p0.01 | 39 200 | 2,3 ms | 54,2 ms | 26,8 ms | 3,9 ms | 1,0 ms | 87,2 ms | 11,3 | 0,6375 |
| wideresnet50 | p0.02 | 78 400 | 2,5 ms | 58,6 ms | 54,2 ms | 0,0 ms | 0,9 ms | 115,3 ms | 8,6 | 0,6395 |
| wideresnet50 | p0.05 | 196 000 | 2,4 ms | 52,7 ms | 133,4 ms | 2,4 ms | 1,0 ms | 190,9 ms | 5,2 | 0,6406 |
| resnet50 | p0.005 | 19 600 | 2,5 ms | 31,3 ms | 16,5 ms | 1,4 ms | 1,0 ms | 51,7 ms | 19,0 | 0,5560 |
| resnet34 | p0.005 | 19 600 | 2,5 ms | 20,1 ms | 16,3 ms | 2,1 ms | 1,0 ms | 40,9 ms | 23,9 | 0,5507 |
| resnet18 | p0.005 | 19 600 | 2,4 ms | 12,9 ms | 13,6 ms | 2,0 ms | 1,0 ms | 30,9 ms | 31,4 | 0,4855 |

- **Le coreset ne débloque plus rien.** Diviser la banque par cinq
  (p0.05 → p0.01) rend 6,1 FPS, la diviser encore par deux n'en rend plus que
  1,6 : le temps faiss suit strictement le nombre de vecteurs, mais le backbone,
  lui, ne bouge pas — ses 54 ms font les trois quarts du budget, et même une
  banque vide plafonnerait vers 17 FPS à 224 px sur wideresnet50.
- **Changer de backbone achète de la cadence et coûte de l'AUROC.** À coreset
  identique, wideresnet50 rend 12,9 FPS pour 0,6321 quand resnet50 en rend 19,0
  pour 0,5560 — la moitié de cadence en plus, un dixième d'AUROC en moins ;
  resnet34 tient encore le niveau de resnet50 (0,5507 pour 23,9 FPS). Les quatre
  banques sont livrées dans `coresets/`, de quoi basculer de l'une à l'autre dans
  la page et voir l'écart en direct.
- **resnet18 est le plancher.** 0,4855, soit sous 0,5 : les FPS qu'il achète ne
  valent plus rien, et les 0,6406 de wideresnet50 restent hors de portée des
  trois autres.

Device : `PATCHCORE_DEVICE` = `auto` (cuda sinon cpu) | `cpu` | `cuda[:N]` | `mps`.
MPS est exclu de l'automatique — PatchCore y échoue sur le pooling adaptatif, et
s'y révèle plus lent que le CPU.

GPU (NVIDIA L40S, `INFER_FAISS_GPU=1`) :

| backbone | coreset | vecteurs en banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | p0.01 | 39 200 | 3,9 ms | 8,2 ms | 0,9 ms | 1,4 ms | 0,7 ms | 14,3 ms | 66,6 | 0,6375 |
| wideresnet50 | p0.02 | 78 400 | 3,9 ms | 8,2 ms | 1,8 ms | 1,4 ms | 0,7 ms | 15,2 ms | 62,7 | 0,6395 |
| wideresnet50 | p0.05 | 196 000 | 4,0 ms | 8,2 ms | 4,1 ms | 1,4 ms | 0,7 ms | 17,7 ms | 54,5 | 0,6406 |

### Le détail par étape

Même banc, découpé plus fin et restreint aux banques que le dépôt livre : 25
mesures par étape après 5 rodages, un processus par banque, frame source
640×480, CPU (torch 4 threads / faiss 1). `backbone` est la passe du réseau
seule, `pooling` ce que `_embed` ajoute derrière — patchify, interpolation,
agrégation — et `post` le déballage des patchs et la segmentation. Les écarts
avec la table de balayage plus haut — au plus 13 %, sur resnet18 — tiennent à
deux campagnes distinctes sur une machine partagée ; les médianes ci-dessous
sont les plus récentes.

| banque | preprocess | backbone | pooling | embed | faiss | post | scoring | encode | total | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 p0.01 | 2,5 ms | 46,9 ms | 7,5 ms | 54,4 ms | 33,1 ms | 2,1 ms | 92,1 ms | 1,0 ms | 93,1 ms | 10,7 | 0,6375 |
| wideresnet50 p0.005 | 2,5 ms | 47,4 ms | 5,9 ms | 53,3 ms | 16,4 ms | 4,5 ms | 76,6 ms | 1,0 ms | 77,6 ms | 12,9 | 0,6321 |
| resnet50 p0.005 | 2,5 ms | 20,7 ms | 8,4 ms | 29,1 ms | 16,4 ms | 2,3 ms | 50,3 ms | 1,0 ms | 51,2 ms | 19,5 | 0,5560 |
| resnet34 p0.005 | 2,5 ms | 17,4 ms | 2,6 ms | 20,1 ms | 16,3 ms | 2,1 ms | 40,9 ms | 1,0 ms | 41,9 ms | 23,9 | 0,5507 |
| resnet18 p0.005 | 2,5 ms | 10,3 ms | 2,4 ms | 12,7 ms | 16,6 ms | 3,8 ms | 35,6 ms | 1,0 ms | 36,5 ms | 27,4 | 0,4855 |

Le backbone domine partout, sauf sur resnet18 où faiss passe devant :

| banque | preprocess | backbone | pooling | faiss | post | encode |
| --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 p0.01 | 2,7 % | 50,4 % | 8,1 % | 35,5 % | 2,2 % | 1,1 % |
| wideresnet50 p0.005 | 3,2 % | 61,1 % | 7,7 % | 21,1 % | 5,8 % | 1,3 % |
| resnet50 p0.005 | 4,9 % | 40,3 % | 16,4 % | 32,0 % | 4,5 % | 1,9 % |
| resnet34 p0.005 | 5,9 % | 41,6 % | 6,3 % | 38,8 % | 5,0 % | 2,3 % |
| resnet18 p0.005 | 6,8 % | 28,1 % | 6,7 % | 45,4 % | 10,3 % | 2,7 % |

Et faiss suit strictement la taille de la banque — 16,4 ms à 19 600 vecteurs,
33,1 à 39 200, soit ×2,02 pour ×2 — sans coût fixe mesurable. Alléger la banque
ne rend donc quelque chose qu'une fois le backbone déjà léger : entre les deux
WideResNet50, diviser la banque par deux rend 2,2 FPS, quand passer à resnet34
en rend 11,0.

Du `.pkg` au PatchCore prêt à scorer :

| banque | extraction (cache froid) | extraction (cache chaud) | `load_bank` | RSS |
| --- | --- | --- | --- | --- |
| wideresnet50 p0.01 | 365,2 ms | 0,04 ms | 976,9 ms | 0,92 Go |
| wideresnet50 p0.005 | 109,5 ms | 0,03 ms | 955,4 ms | 0,84 Go |
| resnet50 p0.005 | 171,1 ms | 0,04 ms | 437,3 ms | 0,66 Go |
| resnet34 p0.005 | 210,0 ms | 0,04 ms | 410,8 ms | 0,61 Go |
| resnet18 p0.005 | 130,2 ms | 0,04 ms | 282,4 ms | 0,54 Go |

L'extraction est mise en cache dans le dossier temporaire du système et n'est
repayée qu'au premier démarrage suivant l'écriture du `.pkg`. Les écarts-types
restent sous 0,3 ms sur faiss et sous 2,7 ms sur le backbone ; la seule étape
franchement dispersée est le `predict` de resnet18 (écart-type 4,2 ms pour une
médiane de 33,1).

## Déploiement sur une scène réelle

Une banque construite sur un dataset public score surtout la nouveauté de scène,
et non l'anomalie cherchée. Sur un robot qui filme toujours le même
environnement, fitter sur *cette* scène : filmer le décor sans l'anomalie à
détecter, sous toutes ses variations, puis envoyer les images dans la moitié
gauche de la page — un dossier `normal/`, et un `anomaly/` si l'on veut de quoi
calibrer un seuil. Tout ce qui n'est pas dans la banque sera scoré comme
anormal, le décor compris s'il a changé.

Quelques centaines d'images espacées valent mieux que des milliers de frames
consécutives : à 30 images par seconde, deux voisines n'apprennent rien de neuf
à la banque.

## Citing

If you use the code in this repository, please cite

```
@misc{roth2021total,
      title={Towards Total Recall in Industrial Anomaly Detection},
      author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
      year={2021},
      eprint={2106.08265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Contact amont pour PatchCore lui-même : karsten.rh1@gmail.com.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
