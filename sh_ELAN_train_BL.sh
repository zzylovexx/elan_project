DATE="0731"
W_PATH="weights/$DATE/"
R_PATH="$DATE/"
DEVICE=2
# hyper-parameter
NORMAL=1 # 0:IMAGENET, 1:ELAN_normal better
GROUP=0
# data path
D_PATH="Elan_3d_box_230808" #"Elan_3d_box"

#FIXED
EPOCH=50
BIN=4
WARMUP=10
# not done yet
COND=0

TRUE=1

W_PATH=$W_PATH"/BL"
R_PATH=$R_PATH"/BL"
PKL=$W_PATH"_B$BIN""_N$NORMAL"
R_PATH=$R_PATH"_B$BIN""_N$NORMAL"
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

#python ELAN_BLtrain.py -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -N=$NORMAL -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH -D_PATH=$D_PATH
python ELAN_EVAL.py -R_PATH=$R_PATH -D_PATH=$D_PATH
echo "SHELL FINISHED"