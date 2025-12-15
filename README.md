[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WG_g337P)
# E4040 2025 Fall Project
## Predictive Modeling for Computational Drug Design
**Project Description**:
We develop predictive models for two key quality metrics in tablet manufacturing: total waste and total impurities. Our study evaluates both traditional machine learning (ML) and modern deep learning (DL) approaches. On the DL side, we investigate three architectures—a 1D convolutional neural network (CNN), a bidirectional long short-term memory network (BiLSTM), and a CNN–BiLSTM hybrid—all trained directly on raw multivariate time-series data, enabling the networks to learn process-relevant temporal patterns without manual feature design. In parallel, we benchmark these models against traditional ML methods trained on post-processed, manually engineered features extracted from the same time-series signals. Among these models, XGBoost and Random Forest achieved the strongest performance. Our results highlight that effective deep learning architectures can eliminate the need for extensive feature engineering while simultaneously demonstrating effective predictive accuracy for critical manufacturing outcomes.

## Project Instructions
Repository for E4040 2025 Fall Project
  - Distributed as Github repository and shared via Github Classroom
  - Contains only `README.md` file

Please read the project instructions carefully. In particular, pay extra attention to the following sections in the project instructions:
 - [Obligatory Github project updates](https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0#bookmark=id.8ga1w2quwf7y)
 - [Student Contributions to the Project](https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0#bookmark=id.3jlnclcqaru7)

The project instructions can be found here:
https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0 

## TODO: This repository is to be used for final project development and documentation, by a group of students
  - Students must have at least one main Jupyter Notebook, and a number of python files in a number of directories and subdirectories such as `utils` or similar, as demonstrated in the assignments
  - The content of this `README.md` should be changed to describe the actual project
  - The organization of the directories has to be meaningful

## Detailed instructions how to submit this project:
1. The project will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. A student's copy of the assignment gets created automatically with a special name
3. **Students must rename the repository per the instructions below**
5. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud
6. If some model is too large to be uploaded to Github - 1) create google (liondrive) directory; 2) upload the model and grant access to e4040TAs@columbia.edu; 3) attach the link in the report and this `README.md`
7. Submit the report as a PDF in the root of this Github repository
8. Also submit the report as a PDF in Courseworks
9. All contents must be submitted to Gradescope for final grading

## TODO: (Re)naming of a project repository shared by multiple students
Students must use a 4-letter groupID, the same one that was chosen in the class spreadsheet in Google Drive: 
* Template: e4040-2025Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2025Fall-Project-MEME-zz9999-aa9999-aa0000.

# Organization of this directory
To be populated by students, as shown in previous assignments.

TODO: Create a directory/file tree
```
e4040-2025Fall-Project-HAHA-kk3764-fnz2101-td2849
├── data
│   ├── Normalization.csv
│   ├── Process.csv
│   └── Time-Series
│       ├── 1.csv
│       ├── 2.csv
│       ├── 3.csv
│       ├── 4.csv
│       ├── 5.csv
│       ├── 6.csv
│       ├── 7.csv
│       ├── 8.csv
│       ├── 9.csv
│       ├── 10.csv
│       ├── 11.csv
│       ├── 12.csv
│       ├── 13.csv
│       ├── 14.csv
│       ├── 15.csv
│       ├── 16.csv
│       ├── 17.csv
│       ├── 18.csv
│       ├── 19.csv
│       ├── 20.csv
│       ├── 21.csv
│       ├── 22.csv
│       ├── 23.csv
│       ├── 24.csv
│       └── 25.csv
├── figures
│   ├── kk3764-fnz2101-td2849_gcp_work_example_screenshot_1.png
│   ├── kk3764-fnz2101-td2849_gcp_work_example_screenshot_2.png
│   └── kk3764-fnz2101-td2849_gcp_work_example_screenshot_3.png
├── model
│   ├── best_bilstm_model
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── best_bilstm_model_weights.h5
│   ├── best_cnn_bilstm_model
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── best_cnn_bilstm_model_weights.h5
│   ├── best_cnn_model
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   └── best_cnn_model_weights.h5
├── utils
│   ├── bilstm.py
│   ├── cnn.py
│   ├── cnn_bilstm.py
│   ├── cnn_lstm.py
│   └── nonlinear.py
├── DL_model.ipynb
├── ML_model_v1.ipynb
├── ML_model_v2.ipynb
└── README.md
```
