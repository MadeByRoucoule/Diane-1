# Diane・1

Diane・1 est une application Python combinant un réseau de neurones simple et une interface graphique moderne pour la reconnaissance de chiffres manuscrits.

## Fonctionnalités

    Réseau de neurones : Architecture [64, 16, 10] pour la classification des chiffres de 0 à 9.
    Interface graphique : Utilisation de customtkinter pour une interaction intuitive.
    Entraînement en temps réel : Suivi de la progression et estimation du temps restant.
    Chargement de poids : Possibilité de charger des poids pré-entraînés pour évaluation.

## Installation

    Clonez le dépôt :
```
git clone https://github.com/MadeByRoucoule/Diane-1.git
cd Diane-1
```

## Installez les dépendances requises :

```
pip install -r requirements.txt
```

## Utilisation

    ### Lancez l'application :

```
python main.py
```

    ### Utilisez l'interface pour :

        Entraîner le modèle sur des données de chiffres manuscrits.
        Tester le modèle sur des échantillons de test.
        Charger des poids sauvegardés pour une évaluation rapide.

## Structure du projet

    main.py : Point d'entrée principal qui initialise le réseau et l'interface.
    network.py : Contient la définition et les fonctions du réseau de neurones.
    interface.py : Gère l'interface utilisateur avec customtkinter.
    data/ : Répertoire pour les données d'entraînement et de test.
    weights/ : Répertoire pour sauvegarder et charger les poids du modèle.

## Contribuer

Les contributions sont les bienvenues ! Veuillez soumettre des issues et des pull requests pour améliorer le projet.
