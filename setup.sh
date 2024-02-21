HOME=/mnt/intel/data/yuwei
CONDA_HOME=$HOME/miniconda3/
CONDA_EXEC=$CONDA_HOME/bin/conda
eval "$($CONDA_EXEC shell.bash hook)"

venv=llava
venv_path=$CONDA_HOME/envs/$venv
if ! [ -d $venv_path ]; then
    conda create -n $venv python=3.10 -y
fi

conda activate $venv
pip install --upgrade pip  # enable PEP 660 support
#git pull
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation

## nuscenes
#pip install nuscenes-devkit

# # Download checkpoints to location outside of the repo
# mkdir -p ckpts
# git clone https://huggingface.co/liuhaotian/llava-v1.5-7b ckpts/llava-v1.5-7b

