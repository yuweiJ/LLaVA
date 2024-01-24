CONDA_HOME=$HOME/miniconda3/
CONDA_EXEC=$CONDA_HOME/bin/conda
eval "$($CONDA_EXEC shell.bash hook)"

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation

git pull
pip install -e .

# # Download checkpoints to location outside of the repo
# mkdir -p ckpts
# git clone https://huggingface.co/liuhaotian/llava-v1.5-7b ckpts/llava-v1.5-7b

