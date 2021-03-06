# Nonparametric-Dirichlet-process-based-Latent-Factor-Model

## citation
If you use the code or datasets in this repository, please acknowledge the paper **"Fine-Grained Job Salary Benchmarking with a Nonparametric Dirichlet-process-based Latent Factor Model"**

## data
You can download the raw datasets used in this paper via the link (cleaned, in csv format, with data descriptions): 

*https://drive.google.com/drive/folders/1eW_30c-ZHYfBip4GAed-DsG9OLqyJZXT?usp=sharing*

You can download the preprocessed data in this paper via the link (in pickle format, for code running purpose): 

*https://drive.google.com/drive/folders/1RsTLFg5tSmESQAkypr5G8-tW48nGMO66?usp=sharing*

Please be aware that if you want to run the model using our datasets directly, you need to download the preprocessed data and put them under the directory: **"src/source_file"** .

## prerequisites (please install the following packages before you run this model NPD-JSB)
- python 3.6
- numpy 1.19.2
- pandas 1.1.3
- scipy 1.5.2

## instructions to run the model NPD-JSB
The src folder includes the source code of the model NPD-JSB.

To run the model, you need to download the whole src folder. Then open a terminal to run the commands as below. According to the different data splitting settings and datasets, there are four different options. I list the four options and the corresponding commands as below.

- option 1 (running the NPD-JSB model on the IT dataset with 5 cross-validations):

 `python main.py --data_split_type cross_validate --data_source it`
 
- option 2 (running the NPD-JSB model on the IT dataset with varying proportional data splittings):

 `python main.py --data_split_type proportional --data_source it`
 
- option 3 (runing the NPD-JSB model on the Finance dataset with 5 cross-validations):

 `python main.py --data_split_type cross_validate --data_source finance`
 
- option 4 ((runing the NPD-JSB model on the Finance dataset with varying proportional data splittings):

 `python main.py --data_split_type proportional --data_source finance`
 
## appendix
The appendix.pdf is the online supplement file for the paper "Fine-Grained Job Salary Benchmarking with a Nonparametric Dirichlet-process-based Latent Factor Model". It includes:

- Proof of the Variational Inference Process,
- Proof of Updating Formulas,
- The Algorithms of Generative and Optimization Process of NDP-JSB,
- Baselines Introduction,
- Statistics and Descriptions of Datasets,
- Additional Experimental Results.

## experimental results
You can access more results from the oringinal paper.
![avatar](https://github.com/qingxin-meng/NDP-JSB/blob/main/figure/5-cross-it.png)
![avatar](https://github.com/qingxin-meng/NDP-JSB/blob/main/figure/5-cross-fin.png)
