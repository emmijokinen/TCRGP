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

Folder *training_data/paper* contains training data files used for the paper. 
Folder *training_data/examples* can be utilized with the example.ipynb
Folder *models* contains pretrained models for different epitopes.
Folder *results* can be used to store result files.

## Updates
TCRGP has been updated in August 20th, 2019, and is not fully compatible with the older version.

## Analysis with TCRGP
Software and data for the single-cell RNA-sequencing analysis of HCC-patients from \[1\] are available at https://github.com/janihuuh/tcrgp_manu_hcc.

# References
\[1\] Emmi Jokinen, Jani Huuhtanen, Satu Mustjoki, Markus Heinonen, and Harri Lähdesmäki. (2019). Determining epitope specificity of T cell receptors with TCRGP. (submitted)

\[2\] Shugay, M. *et al.* (2017). VDJdb: a curated database of T-cell receptor sequences with known antigen specificity. *Nucleic acids research,* **46**(D1), D419-D427

\[3\] Dash, P. *et al.* (2017). Quantifiable predictive features define epitope-specific T cell receptor repertoires. *Nature*, **547**(7661), 89

\[4\] Kawashima S. *et al.* (2007). AAindex: amino acid index database, progress report 2008. *Nucleic Acids Res.*, 36, D202–D205.
