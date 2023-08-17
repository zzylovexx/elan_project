# check every time
DATE="NG_0817"
W_PATH="weights/$DATE"
R_PATH="$DATE"
DEVICE=2
TYPE=0 # 0 dim, 1 angle, 2 both, 3 BL

# hyper-parameter
BIN=4
GROUP=0 #0:NO, 1:cos, 2:sin_sin, 3:compare,
#FIXED
EPOCH=50
WARMUP=10
# not done yet
COND=0
ZERO=0
ONE=1
TWO=2
THREE=3


W_PATH=$W_PATH"/KITTI_"
R_PATH=$R_PATH"/KITTI_"

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
if [ $GROUP != $ZERO ]
then
    PKL=$PKL"_G"$GROUP"_W"$WARMUP
    R_PATH=$R_PATH"_G"$GROUP"_W"$WARMUP
fi
if [ $COND = $ONE ]
then
    PKL=$PKL"_C"
    R_PATH=$R_PATH"_C"
fi
PKL=$PKL"_$EPOCH.pkl"

echo "SHELL W_PATH:"$W_PATH
echo "SHELL PKL:"$PKL
echo "SHELL R_PATH:"$R_PATH
python KITTI_train_all.py -T=$TYPE -W_PATH=$W_PATH -D=$DEVICE -E=$EPOCH -B=$BIN -G=$GROUP -W=$WARMUP -C=$COND
python KITTI_RUN_GT.py -W_PATH=$PKL -R_PATH=$R_PATH -D=$DEVICE
echo "SHELL FINISHED"