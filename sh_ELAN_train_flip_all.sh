DATE="0831_flip"
W_PATH="weights/$DATE/"
R_PATH="$DATE/"
DEVICE=2
TYPE=1 #0:consist dim, 1:angle, 2:both
# hyper-parameter
NORMAL=1 # 0:IMAGENET, 1:ELAN_normal better
GROUP=0
AUGMENT=0
# data path
D_PATH1="Elan_3d_box/"
D_PATH2="Elan_3d_box_230808/"

W_PATH=$W_PATH"Elan_"
R_PATH1=$R_PATH$D_PATH1
R_PATH2=$R_PATH$D_PATH2

#FIXED
EPOCH=50
BIN=4
WARMUP=10
# not done yet
COND=0
ZERO=0
ONE=1
TWO=2


if [ $TYPE = $ZERO ]
then
    W_PATH=$W_PATH"D"
    R_PATH1=$R_PATH1"D"
    R_PATH2=$R_PATH2"D"
fi
if [ $TYPE = $ONE ]
then
    W_PATH=$W_PATH"A"
    R_PATH1=$R_PATH1"A"
    R_PATH2=$R_PATH2"A"
fi
if [ $TYPE = $TWO ]
then
    W_PATH=$W_PATH"DA"
    R_PATH1=$R_PATH1"DA"
    R_PATH2=$R_PATH2"DA"
fi

#W_PATH H-parameters generate in .py
#add to PKL and R_PATH
PKL=$W_PATH"_B$BIN""_N$NORMAL"
R_PATH1=$R_PATH1"_B$BIN""_N$NORMAL"
R_PATH2=$R_PATH2"_B$BIN""_N$NORMAL"
if [ $GROUP = $ONE ]
then
    PKL=$PKL"_G_W$WARMUP"
    R_PATH1=$R_PATH1"_G_W$WARMUP"
    R_PATH2=$R_PATH2"_G_W$WARMUP"
fi
if [ $COND = $ONE ]
then
    PKL=$PKL"_C"
    R_PATH1=$R_PATH1"_C"
    R_PATH2=$R_PATH2"_C"
fi
if [ $AUGMENT = $ONE ]
then
    PKL=$PKL"_aug"
    R_PATH1=$R_PATH1"_aug"
    R_PATH2=$R_PATH2"_aug"
fi
PKL=$PKL"_$EPOCH.pkl"

#python ELAN_Vtrain_flip.py -T=$TYPE -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -N=$NORMAL -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND -A=$AUGMENT
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH1 -D_PATH=$D_PATH1
python ELAN_EVAL.py -R_PATH=$R_PATH1 -D_PATH=$D_PATH1
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH2 -D_PATH=$D_PATH2
python ELAN_EVAL.py -R_PATH=$R_PATH2 -D_PATH=$D_PATH2
echo "SHELL FINISHED"