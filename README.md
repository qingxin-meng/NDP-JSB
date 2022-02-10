# Nonparametric-Dirichlet-process-based-Latent-Factor-Model

If you use the code or datasets in this resporsiry, please acknowedge the paper **"Fine-Grained Job Salary Benchmarking with a Nonparametric Dirichlet-process-based Latent Factor Model"**

You can download the raw datasets used in this paper via the link: *https://drive.google.com/drive/folders/1eW_30c-ZHYfBip4GAed-DsG9OLqyJZXT?usp=sharing*

## prerequisites (please install the following packages before you run this module)
- python 3.6
- numpy 1.19.2
- pandas 1.1.3
- scipy 1.5.2

## intrustions to run the module
The src file include all source code and a source_file folder which include all preprocessed data.
To run the module, you need to download the whole src folder. Then open a terminal to run the commands. According to the different data spliting settings and datasets, there are four different options. I list the four options and the corresponding commands as below.
- option 1 (runing the NPD-JSB model on the IT dataset with 5 cross-validation):
  command: *python main.py --data_split_type cross_validate --data_source it*
- option 2 (runing the NPD-JSB model on the IT dataset with 5 varying proportional data splittings):
  command: *python main.py --data_split_type proportional --data_source it*
- option 3 (runing the NPD-JSB model on the Finance dataset with 5 cross-validation):
  command: *python main.py --data_split_type cross_validate --data_source finance*
- option 4 ((runing the NPD-JSB model on the Finance dataset with 5 varying proportional data splittings):
  command: *python main.py --data_split_type proportional --data_source finance*
