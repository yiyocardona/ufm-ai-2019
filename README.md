# UFM Artificial Intelligence Course - Fall 2019

## Environment Setup
1. [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html?source=post_page---------------------------)
2. `conda create -n ai`
3. `conda activate ai`
4. `conda install pandas scikit-learn matplotlib`
5. `conda install -c conda-forge jupyterlab`
6. [Install PyTorch](https://pytorch.org/) following the specific instructions for your OS. 


## Work Submission Guideline
1. Make a new branch for each deliverable.
2. Commit incremental changes to that branch if applicable (don't add the entire project in a single commit)
3. Make a pull request to master from your branch. If you're using Jupyter notebooks work in a directory where you include the notebook and a plain python file with the same contents as the Jupyter Notebook. Check out [nbconvert](https://nbconvert.readthedocs.io/en/latest/) for an easy way to convert between notebooks and different file formats.
4. Provide the title of the deliverable in the pull request with a small description of the work done.
5. If the work is a single file name it with the following convention: `last_name_first_name_title` otherwise use the same convention for the directory name. Use snakecase for file names.
