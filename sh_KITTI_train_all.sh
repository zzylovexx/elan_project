# check every time
DATE="0925_SGD_ln_iou_02"
W_PATH="weights/$DATE"
R_PATH="$DATE"
DEVICE=2
TYPE=3 # 0 dim, 1 angle, 2 both, 3 BL
NETWORK=0 #0:vgg19_bn, 1:resnet18, 2:densenet121
AUGMENT=1
DEPTH=0
# hyper-parameter
BIN=4
GROUP=0 #0:NO, 1:cos, 2:sin_sin, 3:compare
#FIXED
EPOCH=50
WARMUP=50
# not done yet
COND=0
ZERO=0
ONE=1
TWO=2
THREE=3

W_PATH=$W_PATH"/KITTI_"
R_PATH=$R_PATH"/KITTI_"

#consist_dim, angle, both, Baseline
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
if [ $TYPE = $THREE ]
then
    W_PATH=$W_PATH"BL"
    R_PATH=$R_PATH"BL"
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

PKL=$PKL"_$EPOCH.pkl"
echo "SHELL W_PATH:"$W_PATH
echo "SHELL PKL:"$PKL
echo "SHELL R_PATH:"$R_PATH
python KITTI_train_all.py -T=$TYPE -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND -N=$NETWORK -DEP=$DEPTH -A=$AUGMENT 
python KITTI_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH -D=$DEVICE -N=$NETWORK
echo "SHELL FINISHED"