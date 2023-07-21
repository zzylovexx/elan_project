# check every time
W_PATH="weights/"
R_PATH=""
DEVICE=1

# hyper-parameter
EPOCH=50
BIN=4
NORMAL=1 # 0:IMAGENET, 1:ELAN_normal

# not done yet
GROUP=0
WARMUP=10
COND=0
TRUE=1

PKL=$W_PATH"Vdim_alpha_B$BIN""_N$NORMAL"
R_PATH=$R_PATH"Vdim_alpha_B$BIN""_N$NORMAL"
if [ $GROUP = $TRUE ]
then
    PKL=$PKL"_G_W$WARMUP"
    R_PATH=$R_PATH"_G_W$WARMUP"
fi
if [ $COND = $TRUE ]
then
    PKL=$PKL"_C"
    R_PATH=$R_PATH"_G_W$WARMUP"
fi
PKL=$PKL"_$EPOCH.pkl"
echo $PKL
echo $R_PATH
python ELAN_Vtrain_dim_alpha.py -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -N=$NORMAL -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND;
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH
python ELAN_EVAL.py -R_PATH=$R_PATH
echo "SHELL FINISHED"