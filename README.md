# QSTR
Code for EMNLP2023 paper "A Quality-based Syntactic Template Retriever for Syntactically-controlled Paraphrase Generation".

## Data
We use the QQPPos and ParaNMT datasets for our experiments and retain the same split with [SGCP](https://github.com/malllabiisc/SGCP). You can download the original data from [Data](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ashutosh_iisc_ac_in/ER-roD8qRXFCsyJwbOHOVPgBs-VTKNmkNLzQvM0cLtvBhw?e=a0dOid).

Due to the limitation of file size, I will email you the processed data if you need it. Please contact with me (23111135@bjtu.edu.cn).

## Train the QSTR

```
bash fast-train-roberta.sh
```

## Retrieve templates for Test Set 
```
bash select.sh
```

After obtaining the retrieved templates, you can utilize any syntactically-controlled paraphrase generation models to generate paraphrases.

If you have any questions, please create an issue or email me.
