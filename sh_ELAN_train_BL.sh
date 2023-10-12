DATE="1009_iou"
W_PATH="weights/$DATE/"
R_PATH="$DATE/"
DEVICE=0
# hyper-parameter
IOU=1 # 0:NO, 1:REG alpha, 2:GT alpha (TODO 3:GT dim?)
DEPTH=0 # 0:NO, Calculate with 1:gt_alpha, 2:reg_alpha
NORMAL=1 # 0:IMAGENET, 1:ELAN_normal better
AUGMENT=0
GROUP=0
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

ONE=1
TWO=2

W_PATH=$W_PATH"BL"
R_PATH1=$R_PATH1"BL"
R_PATH2=$R_PATH2"BL"
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
if [ $DEPTH = $ONE ]
then
    PKL=$PKL"_dep"
    R_PATH=$R_PATH"_dep"
fi
if [ $DEPTH = $TWO ]
then
    PKL=$PKL"_depA"
    R_PATH=$R_PATH"_depA"
fi
if [ $IOU = $ONE ]
then
    PKL=$PKL"_iou"
    R_PATH=$R_PATH"_iou"
fi
if [ $IOU = $TWO ]
then
    PKL=$PKL"_iouA"
    R_PATH=$R_PATH"_iouA"
fi
if [ $AUGMENT = $ONE ]
then
    PKL=$PKL"_aug"
    R_PATH1=$R_PATH1"_aug"
    R_PATH2=$R_PATH2"_aug"
fi
PKL=$PKL"_$EPOCH.pkl"

echo "SHELL W_PATH:"$W_PATH
echo "SHELL PKL:"$PKL
echo "SHELL R_PATH1:"$R_PATH1
echo "SHELL R_PATH2:"$R_PATH2
python ELAN_BLtrain.py -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -N=$NORMAL -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND -A=$AUGMENT -IOU=$IOU
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH1 -D_PATH=$D_PATH1
python ELAN_EVAL.py -R_PATH=$R_PATH1 -D_PATH=$D_PATH1
python ELAN_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH2 -D_PATH=$D_PATH2
python ELAN_EVAL.py -R_PATH=$R_PATH2 -D_PATH=$D_PATH2
echo "SHELL FINISHED"
sh ./sh_ELAN_train_BL_2.sh 