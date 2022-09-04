#!/usr/bin/bash
set -x
SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
#SCRIPTPATHCURR ="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
echo VOLUME_SUFFIX
MEM_LIMIT="60g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create autopet_baseline-output-$VOLUME_SUFFIX

echo "Volume created, running evaluation"
 #Do not change any of the parameters to docker run, these are fixed
# --gpus="device=0" \ 
 #       --gpus="all" \
docker run -it --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --gpus="all" \
        -v $SCRIPTPATH/autopet_submission/input/:/input/ \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        autopet_baseline

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t autopet_eval . 
#   -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
docker run --rm -it \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/autopet_submission/expected_output/:/expected_output/ \
        autopet_eval python3 -c """
import SimpleITK as sitk
import os
file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/output_nifti.nii'))
mse = sum(sum(sum((output - expected_output) ** 2)))
if mse == 0.0:
    print('Test passed!')
else:
    print('Test failed!')
"""

docker volume rm autopet_baseline-output-$VOLUME_SUFFIX 
