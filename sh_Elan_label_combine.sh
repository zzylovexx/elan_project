ZZY_PATH="0830_Adam/Elan_3d_box/BL_B4_N1"
RON_PATH="0830_Adam/Elan_3d_box/BL_B4_N1"

echo "RUNNING Label_combine.py"
python Label_combine.py -Z=$ZZY_PATH -R=$RON_PATH
echo "Label_combine.py Done"