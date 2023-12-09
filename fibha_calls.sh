FILES="/media/sd-mmcblk0p2/model_config/*"
for f in $FILES
do
  echo "*************$f*************"
  ./fiba_v2 ./binary_container_1.xclbin 5 $f
done