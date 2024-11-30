# PPGB-MS2

PPGB-MS2 is a tool for predicting the MS/MS of small molecules.

A comparison was made between PPGB-MS2 and two commonly used high-performance secondary mass spectrometry prediction tools, namely 3DMolMS and CFM-ID. The following is the relevant introduction of 3DMolMS and CFM-ID.

## [3DMolMS](https://github.com/JosieHong/3DMolMS)

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. This model's molecular representation, learned through MS/MS prediction tasks, can be further applied to enhance performance in other molecular-related tasks, such as predicting retention times (RT) and collision cross sections (CCS). 

[Read paper in Bioinformatics](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [Try online service at GNPS](https://spectrumprediction.gnps2.org) | [Try model on Konia](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) | [Install from PyPI](https://pypi.org/project/molnetpack/)

🆕 3DMolMS v1.1.10 is now available for inference on **Konia**, **GNPS**, and **PyPI**! 

Please cite: Yuhui Hong, Sujun Li, Christopher J Welch, Shane Tichy, Yuzhen Ye, Haixu Tang, 3DMolMS: prediction of tandem mass spectra from 3D molecular conformations, Bioinformatics, Volume 39, Issue 6, June 2023, btad354, https://doi.org/10.1093/bioinformatics/btad354

## [CFM-ID](https://cfmid.wishartlab.com/)

CFM-ID is a widely used high-performance secondary mass spectrometry prediction tool. It also has a web server available for use. Currently, it has been updated to version 4.0. CFM-ID 4.0 incorporates a more robust machine learning process that utilizes a novel tensor representation to describe the topology of chemical structures. It also adopts a new approach for dealing with ring cleavage and new rule-based methods, which improve the MS/MS spectral prediction for specific classes of chemicals where the machine-learned model performs poorly.

[Read paper in Analytical Chemistry](https://pubs.acs.org/doi/10.1021/acs.analchem.1c01465) | [Read paper in Nucleic Acids Research](https://academic.oup.com/nar/article/50/W1/W165/6591530?login=false)

Please cite:

Wang F, Allen D, Tian S, Oler E, Gautam V, Greiner R, Metz TO, and Wishart DS. (2022) CFM-ID 4.0–a web server for accurate MS-based metabolite identification. Nucleic Acids Research 50 (W1), W165-W174.

Wang F, Liigand J, Tian S, Arndt D, Greiner R, and Wishart D. (2021) CFM-ID 4.0: More Accurate ESI MS/MS Spectral Prediction and Compound Identification. Anal Chem. 93(34):11692-11700.

Djoumbou-Feunang Y, Pon A, Karu N, Zheng J, Li C, Arndt D, Gautam M, Allen F, and Wishart DS. (2019) Significantly Improved ESI-MS/MS Prediction and Compound Identification. Metabolites. 9(4):72.

Allen F, Greiner R, and Wishart DS (2016) Computational prediction of electron ionization mass spectra to assist in GC-MS compound identification. Anal Chem. 88(15):7689-97. Supporting Data

Allen F, Greiner R, and Wishart D (2015) Competitive fragmentation modeling of ESI-MS/MS spectra for putative metabolite identification. Metabolomics. 11:98–110. Supporting Data

Allen F, Pon A, Wilson M, Greiner R, and Wishart DS (2014). CFM-ID: a web server for annotation, spectrum prediction and metabolite identification from tandem mass spectra. Nucleic Acids Res. 42(Web Server issue):W94-9.