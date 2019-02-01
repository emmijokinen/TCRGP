# TCRGP
TCRGP is a novel Gaussian process method that can predict if TCRs recognize certain epitopes. This method can utilize different CDR sequences from both TCRα and TCRβ chains from single-cell data and learn which CDRs are important in recognizing the different epitopes. TCRGP has been developed at Aalto University.

* For a comprehensive description of TCRGP see \[1\]
* For examples of usage, see Examples.ipynb

## Dependencies
To use TCRGP, you will need to have
* [TensorFlow](https://www.tensorflow.org/) (We have used version 1.8.0)
* [GPflow](http://gpflow.readthedocs.io/) (We have used version 1.1.1)
* And some other Python packages, which are imported at the beginning of tcrgp.py

## Data
The data in folder *data* has been obtained from \[2\], \[3\] and \[4\].

Other folders (*training_data, models, results*) and their contents exist for demonstrative purposes and they are utilized by Example.ipynb.


# References
\[1\] Emmi Jokinen, Markus Heinonen, Jani Huuhtanen, Satu Mustjoki and Harri Lähdesmäki. (2018). TCRGP: Determining epitope specificity of T cell receptors. (submitted)

\[2\] Shugay, M. *et al.* (2017). VDJdb: a curated database of T-cell receptor sequences with known antigen specificity. *Nucleic acids research,* **46**(D1), D419-D427

\[3\] Dash, P. *et al.* (2017). Quantifiable predictive features define epitope-specific T cell receptor repertoires. *Nature*, **547**(7661), 89

\[4\] Kawashima S. *et al.* (2007). AAindex: amino acid index database, progress report 2008. *Nucleic Acids Res.*, 36, D202–D205.
