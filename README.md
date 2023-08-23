Boosting Neural Network Performance for scRNA-seq Data through Random Projections
=================================================================================

This repository contains the code and data related to the paper titled "Boosting Neural Network Performance for scRNA-seq Data through Random Projections." The paper explores the use of random projections to enhance the performance of neural networks when dealing with single-cell RNA sequencing (scRNA-seq) data. This Markdown file serves as a guide to the repository structure and provides placeholders for additional information that will be added.

Usage
-----

For the execution of the source code, you simply need to execute the `execute.py` file. There are no needed parameters for the execution of the file. It automatically executes all the experiments and saves the results in the `scores` folder. The results are saved in `.pkl` files, and the name of the file is the name of the dataset it comes from. The `.pkl` files are compressed pickle-dump files. Each file is a dictionary with the keys the augmentation rate $K$, and the values are dictionaries. The subsequent dictionaries have keys the models' names and values the models' scores averaged over 100 runs.

Finally, in the `labels` folder are the labels of the predictions executed by the scNNBoost (nn_maj) model and the RF-Majority (rf_maj) model. The labels are saved in `.pkl` files, and the name of the file is the name of the dataset it comes from. The `.pkl` files are compressed pickle-dump files. Each file is a dictionary with keys the augmentation rates $K$, and the values are dictionaries. The subsequent dictionaries have keys the models' names and values the labels of the predictions of the model. The labels are np.ndarrays of shape (n_samples * 0.1, 200). The columns are separated into groups of 2, where the first column is the index of the sample, and the second column is the label of the prediction.


Citation
--------

If you use this paper, code, or data in your work, please cite:

TODO: Add the citation information for the paper once available.

Acknowledgments
---------------

Financed by the European Union - NextGenerationEU through Recovery and Resilience Facility, Greece 2.0, under the call RESEARCH – CREATE – INNOVATE (project code:TAEDK-06185 / Τ2EDK- 02800)

---

For any inquiries or issues related to this repository, please contact [Panagiotis Anagnostou](mailto:panagno@uth.gr), or create an issue in the repository.
