# check every time
DATE="0731"
W_PATH="weights/$DATE/"
R_PATH="$DATE/"
DEVICE=2
TYPE=2
# hyper-parameter
NORMAL=1 # 0:IMAGENET, 1:ELAN_normal better
GROUP=1

#FIXED
EPOCH=50
BIN=4
WARMUP=10
# not done yet
COND=0
ZERO=0
ONE=1
TWO=2

mkdir $W_PATH
if [ $TYPE = $ZERO ]
then
    W_PATH=$W_PATH"D"
    R_PATH=$R_PATH"D"
fi
if [ $TYPE = $ONE ]
then
    W_PATH=$W_PATH"A"
    R_PATH=$R_PATH"A"
fi
if [ $TYPE = $TWO ]
then
    W_PATH=$W_PATH"DA"
    R_PATH=$R_PATH"DA"
fi

PKL=$W_PATH"_B$BIN""_N$NORMAL"
R_PATH=$R_PATH"_B$BIN""_N$NORMAL"
if [ $GROUP = $ONE ]
then
    PKL=$PKL"_G_W$WARMUP"
    R_PATH=$R_PATH"_G_W$WARMUP"
fi
if [ $COND = $ONE ]
then
    PKL=$PKL"_C"
    R_PATH=$R_PATH"_G_W$WARMUP"
fi
PKL=$PKL"_$EPOCH.pkl"
echo $PKL
echo $R_PATH

python ELAN_Vtrain_all.py -T=$TYPE -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -N=$NORMAL -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND;
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH
python ELAN_EVAL.py -R_PATH=$R_PATH
echo "SHELL FINISHED"