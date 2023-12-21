# 🪐 NLP Project: Data-Driven taxonomy for cognitive distortion

This repository holds the code for the project for Natural Language Processing (S2023). It contains a script called data-driven-taxonomy which allows for interactive exploration of a cluster of negative thought.

## 📋 Review

### ⏭ Workflows

### 🗂 Bibliography

### ⏭ Workflows

### 🗂 Assets

## Description of the data
To demonstrate the functionality of the package as collection of lyrics from danish songs from 10 artists were scraped from Genius.com. Up to 5 songs from each artist were scraped, but only the danish songs were saved. 

| thought | original_label |
| ----------------- | -: |
| Someone I trusted stole something valuable of mine, I was extremely angry and wanted justice       | emotional reasoning |
| She doesn't respect me.        | overgeneralizing |
| **Total**         | **921**|

## Usage and reproducibility

The code was developed and tested on a MacBook Pro with macOS (Sonoma 14.1.2, python v3.9.6).

To reproduce the results , follow the steps below. All terminal commands should be run from the root directory of the repository.


1. Clone the repository
2. Create a virtual environment and install requirements
```
bash setup.sh
```
3. Run the `run.sh` script to: 
    - Scrape danish song lyrics from Genius
    - Preprocess the data and prepare dataframe

## Repository structure
```
```

## Results
To display the results of the example functionality of the package using the danish songs, follow the following links:
- [Cluster](http://htmlpreview.github.io/?https://https://github.com/SylvainEstebe/cognitive_distortion_project/blob/main/export/cluster_manual2_all-MiniLM-L12-v2.html)

These can also be found in the `examples` folder of the repository.
