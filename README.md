# QSTR
Code for EMNLP2023 paper "A Quality-based Syntactic Template Retriever for Syntactically-controlled Paraphrase Generation".

## Data
We use the QQPPos and ParaNMT datasets for our experiments and retain the same split with [SGCP](https://github.com/malllabiisc/SGCP). You can download the data from [Data](https://drive.google.com/file/d/1fovvaKD6N2FssVl1lCsS-NrnI8mmygah/view?usp=drive_link).

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
