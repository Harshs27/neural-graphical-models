# Update conda environment.
conda update -n base conda;
conda update --all;

# Create conda environment.
conda create -n ngm python=3.8 -y;
conda activate ngm;
conda install -c conda-forge notebook -y;
python -m ipykernel install --user --name ngm;

# install pytorch (1.9.0 version)
conda install numpy -y;
# conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y;
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y;

# Install packages from conda-forge.
conda install -c conda-forge matplotlib -y;

# Install packages from anaconda.
# conda install -c anaconda pandas networkx scipy -y;
# Alternate to anaconda channel
conda install -c conda-forge pandas networkx scipy -y;

# Install pygraphviz (Optional)
conda install --channel conda-forge graphviz pygraphviz -y;

# Install pip packages
pip3 install -U scikit-learn;

# Install packages from pip. (Optional)
pip install pyvis;
pip install --upgrade scipy networkx;

# Create environment.yml.
conda env export > environment.yml;
