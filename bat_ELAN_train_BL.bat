@echo off
set DATE=1012_depth
set "W_PATH=weights/%DATE%/"
set "R_PATH=%DATE%"
set DEVICE=0
    rem hyper-parameter
set IOU=0
    rem 0:NO, 1:REG alpha, 2:GT alpha (TODO 3:GT dim?)
set DEPTH=1
    rem 0:NO, Calculate with 1:REG alpha,, 2:GT alpha 
set NORMAL=1
    rem 0:IMAGENET, 1:ELAN_normal better
set AUGMENT=0
set GROUP=0
    rem data path
set D_PATH1=Elan_3d_box/
set D_PATH2=Elan_3d_box_230808/

set "W_PATH=%W_PATH%Elan_"
set "R_PATH1=%R_PATH%/%D_PATH1%"
set "R_PATH2=%R_PATH%/%D_PATH2%"

    rem FIXED
set EPOCH=50
set BIN=4
set WARMUP=10
    rem not done yet
set COND=0

set ONE=1
set TWO=2

set "W_PATH=%W_PATH%BL"
set "R_PATH1=%R_PATH1%BL"
set "R_PATH2=%R_PATH2%BL"
set "PKL=%W_PATH%_B%BIN%_N%NORMAL%"
set "R_PATH1=%R_PATH1%_B%BIN%_N%NORMAL%"
set "R_PATH2=%R_PATH2%_B%BIN%_N%NORMAL%"

if %GROUP% equ %ONE% (
    set "PKL=%PKL%_G_W%WARMUP%"
    set "R_PATH1=%R_PATH1%_G_W%WARMUP%"
    set "R_PATH2=%R_PATH2%_G_W%WARMUP%"
)
if %COND% equ %ONE% (
    set "PKL=%PKL%_C"
    set "R_PATH1=%R_PATH1%_C"
    set "R_PATH2=%R_PATH2%_C"
)
if %DEPTH% equ %ONE% (
    set "PKL=%PKL%_dep"
    set "R_PATH1=%R_PATH1%_dep"
    set "R_PATH2=%R_PATH2%_dep"
)
if %DEPTH% equ %TWO% (
    set "PKL=%PKL%_depA"
    set "R_PATH1=%R_PATH1%_depA"
    set "R_PATH2=%R_PATH2%_depA"
)
if %IOU% equ %ONE% (
    set "PKL=%PKL%_iou"
    set "R_PATH1=%R_PATH1%_iou"
    set "R_PATH2=%R_PATH2%_iou"
)
if %IOU% equ %TWO% (
    set "PKL=%PKL%_iouA"
    set "R_PATH1=%R_PATH1%_iouA"
    set "R_PATH2=%R_PATH2%_iouA"
)
if %AUGMENT% equ %ONE% (
    set "PKL=%PKL%_aug"
    set "R_PATH1=%R_PATH1%_aug"
    set "R_PATH2=%R_PATH2%_aug"
)
set "PKL=%PKL%_%EPOCH%.pkl"

echo "BAT W_PATH:%W_PATH%"
echo "BAT PKL:%PKL%"
echo "BAT R_PATH1:%R_PATH1%"
echo "BAT R_PATH2:%R_PATH2%"
python ELAN_BLtrain.py -W_PATH=%W_PATH% -D=%DEVICE% -E=%EPOCH% -N=%NORMAL% -B=%BIN% -G=%GROUP% -W=%WARMUP% -C=%COND% -A=%AUGMENT% -DEP=%DEPTH% -IOU=%IOU%
python ELAN_RUN_GT.py -W_PATH=%PKL% -R_PATH=%R_PATH1% -D_PATH=%D_PATH1%
python ELAN_EVAL.py -R_PATH=%R_PATH1% -D_PATH=%D_PATH1%
python ELAN_RUN_GT.py -W_PATH=%PKL% -R_PATH=%R_PATH2% -D_PATH=%D_PATH2%
python ELAN_EVAL.py -R_PATH=%R_PATH2% -D_PATH=%D_PATH2%
echo "BAT FINISHED"
call bat_ELAN_train_BL_2.bat