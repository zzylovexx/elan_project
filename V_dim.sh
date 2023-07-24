# check every time
DATE="0724"
W_PATH="weights/$DATE/V_dim"
R_PATH="$DATE/V_dim"
DEVICE=2
# hyper-parameter
NORMAL=0 # 0:IMAGENET, 1:ELAN_normal
GROUP=1

#FIXED
EPOCH=50
BIN=4
WARMUP=10
# not done yet
COND=0

TRUE=1

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
echo $PKL
echo $R_PATH
python ELAN_Vtrain_dim.py -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -N=$NORMAL -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND;
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH
python ELAN_EVAL.py -R_PATH=$R_PATH
echo "SHELL FINISHED"