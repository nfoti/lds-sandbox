
if [[ ! $(conda info --envs | cut -d" " -f1 | grep -E lds-sandbox) ]]
then
    echo "creating conda environment lds-sandbox"
    conda-env create -f environment.yml
    source activate lds-sandbox
    rm -rf deps/*
    pip install --src deps -e git+ssh://git@github.com/nfoti/autograd_linalg.git#egg=autograd_linalg
    pip install --src deps -e git+ssh://git@github.com/jackkamm/einsum2.git#egg=einsum2
else
    echo "lds-sandbox environment already exists"
fi

