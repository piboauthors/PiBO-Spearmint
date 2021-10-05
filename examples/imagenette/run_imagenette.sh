# bash run_imagenette.sh 8e-3 0.99 0.95 1e-6 0.4 MaxPool 
source $HOME/.conda/bin/activate fastai
MKL_SERVICE_FORCE_INTEL=1 python $HOME/spearmint_priors/examples/imagenette/train_imagenette.py --lr $1 --sqrmom $2 --mom $3 --eps $4 --bs 64 --opt ranger --sa --fp16 --arch xse_resnext50 --mixup $5 --pool $6 --epochs 80 --size 128
