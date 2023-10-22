# check every time
DATE="1022_trun_mse"
W_PATH="weights/$DATE"
R_PATH="$DATE"
#L_PATH="weights/1020/KITTI_BL_B4_dep_iou_vgg_best.pkl"
L_PATH=""

DEVICE=0
# hyper-parameter
TYPE=0 # 0:BL, 1:C_dim, 2:C_angle, 3 C_Both
IOU=0 # 0:NO, 1:REG alpha (iou), 2:GT alpha (iouA) [TODO] 3:GT dim?
DEPTH=0 # 0:NO, 1:REG alpha (dep), 2: GT alpha (depA)
GROUP=0 # 0:NO, 1:cos, 2:sin_sin, 3:compare

#FIXED
AUGMENT=0
EPOCH=50
WARMUP=0
NETWORK=0 # 0:vgg19_bn, 1:resnet18, 2:densenet121
BIN=4
# not done yet
COND=0
#
ZERO=0
ONE=1
TWO=2
THREE=3

W_PATH=$W_PATH"/KITTI_"
R_PATH=$R_PATH"/KITTI_"

#Baseline, consist_dim, consist_angle, both, 
if [ $TYPE = $ZERO ]
then
    W_PATH=$W_PATH"BL"
    R_PATH=$R_PATH"BL"
fi
if [ $TYPE = $ONE ]
then
    W_PATH=$W_PATH"D"
    R_PATH=$R_PATH"D"
fi
if [ $TYPE = $TWO ]
then
    W_PATH=$W_PATH"A"
    R_PATH=$R_PATH"A"
fi
if [ $TYPE = $THREE ]
then
    W_PATH=$W_PATH"DA"
    R_PATH=$R_PATH"DA"
fi

#W_PATH H-parameters generate in .py
#add to PKL and R_PATH
PKL=$W_PATH"_B$BIN"
R_PATH=$R_PATH"_B$BIN"
#is_group
if [ $GROUP != $ZERO ]
then
    PKL=$PKL"_G"$GROUP"_W"$WARMUP
    R_PATH=$R_PATH"_G"$GROUP"_W"$WARMUP
fi
#condition
if [ $COND = $ONE ]
then
    PKL=$PKL"_C"
    R_PATH=$R_PATH"_C"
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
    R_PATH=$R_PATH"_aug"
fi
#network
if [ $NETWORK = $ZERO ]
then
    PKL=$PKL"_vgg"
    R_PATH=$R_PATH"_vgg"
fi
if [ $NETWORK = $ONE ]
then
    PKL=$PKL"_resnet"
    R_PATH=$R_PATH"_resnet"
fi
if [ $NETWORK = $TWO ]
then
    PKL=$PKL"_dense"
    R_PATH=$R_PATH"_dense"
fi

#PKL=$PKL"_$EPOCH.pkl"
PKL=$PKL"_best.pkl"
echo "SHELL W_PATH:"$W_PATH
echo "SHELL PKL:"$PKL
echo "SHELL R_PATH:"$R_PATH
python KITTI_train_trun.py -T=$TYPE -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND -N=$NETWORK -DEP=$DEPTH -IOU=$IOU -A=$AUGMENT -L_PATH=$L_PATH
python KITTI_RUN_GT_trun.py -W_PATH=$PKL -R_PATH=$R_PATH -D=$DEVICE -N=$NETWORK
echo "SHELL FINISHED"