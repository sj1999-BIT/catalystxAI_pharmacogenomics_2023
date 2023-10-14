# catalystxAI_pharmacogenomics_2023
A transformer-based model for predicting cancer drug response

Identification of new lead molecules for treatment  of cancer is often the result of more than a decade of dedicated efforts. Before these painstakingly selected drug candidates are used in clinic , their anti-cancer activity was generally varified by vitro cell experiments. Therefore, accurately prediction  cancer drug response is the key and challenging task in anti-cancer drugs design and precision medicine. With the development of pharmacogenomics, combining efficient methods of extracting drug features and omic data makes it possible to improve drug response prediction. We utilise DeepTTC, an novel  end-to-end deep learning model to predict the anti-cancer drug response. DeepTTC takes gene expression data and chemical strcutures  of drug for drug response prediction. Specifically, get inspiration from natural language processing, DeepTTC use  transformer for drug representation learning. Compared to existing methods, DeepTTC achieved higher performance in terms of RMSE, Pearson correlation coefficient and Spearman's rank correlation coefficient on multiple test sets. Moreover, we explorated  identify multiple clinical  indications anti-cancer drugs. With the outstanding preformance, DeepTTC is expected to the  most effective in silico  measure  in cancer drug design.


## Install

```          
```


## Data description
https://academic.oup.com/bib/article/23/3/bbac100/6554594


## Run Step

```
step1
# Get the Drug SMILES information in the NCBI pubchem database based on the pubchem id from GDSC provided.
python Step1_PubchemID2smile.py GDSC_data/Drug_listTue_Aug10_2021.csv 


step2 
# This is a python Class script to get the cell line RNA-seq data base the "drug-id and cosmic_id".
python Step1_getData.py


step3 
# The DeepTTC model script, includ Data split and model train/test.
python Step3_model.py

```

## Paper
Jiang, Likun, et al. "DeepTTA: a transformer-based model for predicting cancer drug response." Briefings in Bioinformatics 23.3 (2022): bbac100.
