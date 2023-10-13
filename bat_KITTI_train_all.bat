@echo off
set DATE=1012_iou
set "W_PATH=weights/%DATE%"
set "R_PATH=%DATE%"
set DEVICE=0
    rem 0 dim, 1 angle, 2 both, 3 BL
set TYPE=3
    rem 0:NO, 1:REG alpha (iou), 2:GT alpha (iouA) [TODO] 3:GT dim?
set IOU=1
    rem 0:NO, Calculate with 1:REG alpha (dep), 2: GT alpha (depA)
set DEPTH=0 
set AUGMENT=0
    rem hyper-parameter
    rem 0:vgg19_bn, 1:resnet18, 2:densenet121
set NETWORK=0 
set BIN=4
    rem 0:NO, 1:cos, 2:sin_sin, 3:compare
set GROUP=0 

set EPOCH=50
set WARMUP=50
    rem not done yet
set COND=0

set ZERO=0
set ONE=1
set TWO=2
set THREE=3

set "W_PATH=%W_PATH%/KITTI_"
set "R_PATH=%R_PATH%"/KITTI_"

    rem consist_dim, angle, both, Baseline
if %TYPE% equ %ZERO% (
    set "W_PATH=%W_PATH%D"
    set "R_PATH=%R_PATH%D"
)
if %TYPE% equ %ONE% (
    set "W_PATH=%W_PATH%A"
    set "R_PATH=%R_PATH%A"
)
if %TYPE% equ %TWO% (
    set "W_PATH=%W_PATH%DA"
    set "R_PATH=%R_PATH%DA"
)
if %TYPE% equ %THREE% (
    set "W_PATH=%W_PATH%BL"
    set "R_PATH=%R_PATH%BL"
)

rem W_PATH H-parameters generate in .py #add to PKL and R_PATH

set "PKL=%W_PATH%_B%BIN%"
set "R_PATH=$R_PATH_B%BIN%"
    rem is_group
if %GROUP% neq %ZERO% (
    set "PKL=%PKL%_G%GROUP%_W%$WARMUP%"
    set "R_PATH=%R_PATH%_G%GROUP%_W%WARMUP%"
)
    rem condition
if %COND% equ %ONE% (
    set "PKL=%PKL%_C"
    set "R_PATH=%R_PATH%_C"
)
if %DEPTH% equ %ONE% (
    set "PKL=%PKL%_dep"
    set "R_PATH=%R_PATH%_dep"
)
if %DEPTH% equ %TWO% (
    set "PKL=%PKL%_depA"
    set "R_PATH=%R_PATH%_depA"
)
if %IOU% equ %ONE% (
    set "PKL=%PKL%_iou"
    set "R_PATH=%R_PATH%_iou"
)
if %IOU% equ %TWO% (
    set "PKL=%PKL%_iou"
    set "R_PATH=%R_PATH%_iou"
)
if %AUGMENT% equ %ONE% (
    set "PKL=%PKL%_aug"
    set "R_PATH=%R_PATH%_aug"
)
    rem network
if %NETWORK% equ %ZERO% (
    set "PKL=%PKL%_vgg"
    set "R_PATH=%R_PATH%_vgg"
)
if %NETWORK% equ %ONE% (
    set "PKL=%PKL%_resnet"
    set "R_PATH=%R_PATH%_resnet"
)
if %NETWORK% equ %TWO% (
    set "PKL=%PKL%_dense"
    set "R_PATH=%R_PATH%_dense"
)

set "PKL=%PKL%_%EPOCH%.pkl"
echo "SHELL W_PATH:%W_PATH%"
echo "SHELL PKL:%PKL%"
echo "SHELL R_PATH:%R_PATH%"
python KITTI_train_all.py -T=%TYPE% -W_PATH=%W_PATH% -D=%DEVICE% -E=%EPOCH% -B=%BIN% -G=%GROUP% -W=%WARMUP% -C=%COND% -N=%NETWORK% -DEP=%DEPTH% -IOU=%IOU% -A=%AUGMENT% 
rem python KITTI_RUN_GT.py -W_PATH=%PKL% -R_PATH=%R_PATH% -D=%DEVICE% -N=%NETWORK%
echo "SHELL FINISHED"