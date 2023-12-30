🪐 NLP Project: Data-Driven taxonomy for cognitive distortion

This repository holds the code for the project for Natural Language Processing (S2023). It contains a script called data-driven-taxonomy which allows for interactive exploration of a cluster of negative thought.

## 📋 Review

### ⏭ Workflows

![image](https://github.com/SylvainEstebe/cognitive_distortion_project/assets/75991758/e101cc02-f95c-46ee-94cf-69fa1af4eca3)# 

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
├── code 
│   ├── 
│   ├── 
├── env                                         <- Not included in repo
├── data
│   ├── src
│   ├── src
├── export                                   
│   ├──
│   ├── 
│   ├── 
│   └── 
├── .gitignore
├── README.md
├── requirements.txt
```
https://github.com/SylvainEstebe/cognitive_distortion_project/blob/main/export/aglom_All-Distilroberta-v1.html
## Results
To display the results of the example functionality of the package using the danish songs, follow the following links:
- [k-mean](http://htmlpreview.github.io/?https://github.com/SylvainEstebe/cognitive_distortion_project/blob/main/export/k_mean_All-Distilroberta-v1.html)
- [aglomerative](http://htmlpreview.github.io/?https://github.com/SylvainEstebe/cognitive_distortion_project/blob/main/export/aglom_All-Distilroberta-v1.html)
- [hdbscan](http://htmlpreview.github.io/?https://github.com/SylvainEstebe/cognitive_distortion_project/blob/main/export/hdbscan_manualAll-Distilroberta-v1.html)
- [embedding](http://htmlpreview.github.io/?https://github.com/SylvainEstebe/cognitive_distortion_project/blob/main/export/embeddingall-MiniLM-L12-v2.html)

These can also be found in the `examples` folder of the repository.

### 🗂 Bibliography
