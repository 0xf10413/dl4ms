A Deep Learning Framework for Motion Synthesis and Editing
==========================================================

Documents originaux
-------------------
Les dossiers suivants faisaient partie du code original :
- data : données brutes (BVH, ...)
- nn : fichiers python pour les réseaux de neurones
- motion : fichiers python pour la gestion des données de mouvement
- synth : fichiers python de démonstration et d'apprentissage

À quelques exceptions près, les fichiers originaux sont restés.
Consulter hier.txt pour plus de détail sur la hiérarchie originale.

blender
-------
Quelques scripts pour importer des données de mouvement dans blender.
J'ai laissé uniquement ceux qui me servaient le plus souvent et que je retouchais
à chaque fois.

Le format des données varie légèrement en fonction de ce qu'on veut importer, donc des
retouches sont nécessaires. Ceci dit, la représentation exacte
du squelette ne varie pas. Il y a donc des infos utiles à récupérer,
notamment sur le "comment transformer le .npz en un squelette"

curve_editor
------------
Une tentative de démonstrateur pour le logiciel. Donne une petite UI pour créer
une courbe, puis générer un mouvement par dessus.

Dépendances :
-> qt5.7 (ou plus)
-> boost-python (et la version de python3 correspondante)
Attention, boost-python est difficile à configurer avec python3 sous windows
(mais pas impossible)

L'UI est omposée de quatre parties :
A B C
 D

A : Visionneuse 3D du squelette et de sa trajectoire. Commandes à la minecraft,
  une fois le focus obtenu, zqsd pour se déplacer en avant, à gauche, à droite,
  en arrière, shift pour descendre, space pour monter. Comandes simplifiées :
  clic droit pour reculer, bouton milieu pour avancer. Une fois le focus obtenu,
  la caméra peut être tournée avec la souris.
B : Interpréteur python. Il est possible d'y mettre n'importe quelle commande
    python valide, imports inclus. L'état est sauvegardé entre les exécutions.
    Certaines variables ont une signification particulière et représentent
    des objets dans la vue 3D : skel (le squelette, taille nb frame × 22 × 3),
    curve (la courbe de déplacement à suivre, 3 × nb frames)
    et skel_parents (les parents des 22 articulations). Le code s'active en appuyant
    sur ctrl+entrée. Un morceau de code de démonstration est déjà affiché.
C : des morceaux de codes à lancer séquentiellement pour construire un squelette
    suivant la trajectoire. Ils permettent séquentiellement de : charger le
    footstepper (réseau de pré-calcul pour trouver le "rythme des pas"), charger
    le gros réseau principal (qui va générer un premier mouvement), charger
    la fonction de contrainte (qui va corriger le mouvement généré) et appliquer
    les contraintes (qui lance le calcl effectif). Il s'agit des morceaux
    de code python du dossier rc/. À noter qu'ils modifient les variables
    existantes dans l'interpréteur.
D : la sortie de l'interpréteur. Les print() sont capturés, ainsi que les exceptions.

En éditant directement curve dans l'interpréteur python, il devient possible
de spécifier une nouvelle tranjectoire. Attention : le format est un peu spécial !
La courbe contient les données de déplacement sur l'axe x, sur l'axe y, et de rotation
au format différentiel. En d'autres termes, chaque triplet (dx, dy, domega) est
un **déplacement** par rapport à la frame précédente. Attention également, le
troisième paramètre est une rotation selon l'axe vertical, et non pas un déplacement
vertical. Enfin, dx est à utiliser assez rarement -- il donne des pas de côté
peu réalistes.

Gram trainer
-----------
Les fichiers d'entraînement des classificateurs SVM et SLP sur les données.

- demo_print_database.py ; affiche quelques données de mouvement remarquables
  dans les bdd, et les différences mutuelles mesurables entre les mouvements.
  Il a été utilisé au début, à l'occasion de quelques réunions.
- print_massive_gram.py : comme le précédent, récupère certains mouvements
  des bdd et affiche des données à leur sujet. Contrairement au précédent,
  affiche les __matrices de gram__ des mouvements. Utilisé pour observer
  les différences entre matrices de Gram.
- show_fourier.py : comme le précédent, sauf qu'il affiche les parties réelles
  et imaginaires des données encodées.
- train_all.py : gros script qui fait tout le travail de classification des données,
  pour le SVM et le SLP. Crée un dossier cache dans lequel il stocke autant de
  résultats intermédiaires que possible. Les transformations appliquées sont
  fonction du premier argument en ligne de commande : s'il contient "orig" ou "direct",
  les données ne seront pas encodées ; s'il contient "caché", les données seront
  encodées ; l'un de ces trois termes doit apparaître, si plusieurs de ces
  termes apparaissent, seul le premier sera pris en compte. S'il contient "gram",
  la classification se fera sur la matrice de Gram ; "nothing", sur les données
  sans traitement supplémentaire ; "fourier", les coefficients de fourier en
  forme algébrique ; "fourier" et "abs" : coefficients de fourier en amplitude
  uniquement ; "fourier" et "abs" et "phase", coefficients de fourier en amplitude
  et phase.

Les résultats de la classification seront donnés dans le dossier de cache,
dans un sous-dossier de nom le premier argument en ligne de commande. L'accuracy
est directement calculée, dans un fichier type "slp_motions_accuracy.txt".

La classification se fait sur les données de style ET les données de mouvement.
En plus des données connues, quelques essais supplémentaires sont faits sur
edin_punching : un csv généré donne la liste des mouvements/styles attendus
et générés.

Le script train_all est conçu autour du cacher, dont le fonctionnement
est un peu spécial. Celui-ci a une méthode retrieve qui prend en argument
un nom de fichier et une méthode de fabrication de ce fichier (sous forme
de fonction). Si le fichier existe déjà, son contenu est renvoyé (les
pré-traitements à appliquer dépendent du format annoncé), sinon, la méthode
de construction est appelée (et le Cacher vérifie que le fichier a bien
été créé).

Au moment de la création des données d'apprentissage, 75% des données
sont gardées pour l'apprentissage, le reste est mis de côté pour les tests.
Pour assurer une répartition équitable, on a bien 75% de chaque mouvement
et 75% de chaque style présent dans les données d'apprentissage.

gan_trainer
----------
Fichiers d'entraînement de GAN. Il y a deux groupes de fichiers :
ceux travaillant sur les données de mouvement, et ceux travaillant
sur des données mono-dimensionnelles.

Données mono-dimensionnelles :
- distributions.py : quelques distributions aléatoires
- gasse_gan.py : fichier de GAN écrit par Maxime Gasse, inspiré d'un de mes scripts
- gasse_wgan.py : idem, mais avec le WGAN
- simplegan.py : implémentation refactorisée de gasse_gan.py
- wgan.py : idem, mais avec wgan
- main.py : fichier de lancement général
Ce set de fichiers m'a servi à me "faire la main" sur les GANs.

Données de mouvement :
- autoencodeur.py : des tentatives de réimplémenter l'auto-encodeur siggraph de
  [Holden, 2015] et  [Holden, 2016]. N'ont pas fonctionné, à la place j'ai changé
  directement leur implémentation.
- cyclegan_dcgan.py : la version cyclegan du GAN basé dcgan, voir plus bas.
- cyclegan.py : la version cyclegan du GAN basé gram, voir plus bas.
- dcgan.py : GAN basé dcgan pour la génération de mouvements.
- real_gan_first.py : 
- real_gan.py : fichier de travail pour tester des architectures. Il n'est pas à jour.
- real_gan_simpler.py : 
- real_gan_wdl4ms.py
- real_gan_wdl4ms_unfact.py
- real_gan_working.py

 
