#FROM python:3.10.4 #change by Max/Lars  
FROM python:3.9  

run groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

#run mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
#    && chown algorithm:algorithm /opt/algorithm /input /output

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

user algorithm

workdir /opt/algorithm

env path="/home/algorithm/.local/bin:${path}"

run python -m pip install --user torch==1.11.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

copy --chown=algorithm:algorithm weights.zip /opt/algorithm/
run python -c "import zipfile; zipfile.ZipFile('/opt/algorithm/weights.zip').extractall('/opt/algorithm/weights')"

copy --chown=algorithm:algorithm requirements.txt /opt/algorithm/
run python -m pip install --user -rrequirements.txt

add --chown=algorithm:algorithm models /opt/algorithm/models
run mkdir /opt/algorithm/monai_input

copy --chown=algorithm:algorithm    infer.py \
                                    infer_swin.py \
                                    infer_class.py \
                                    ap_baseline_nnunet_predict.py \
                                    ap_baseline_nnunet_process.py \
                                    /opt/algorithm/


# nnunet specific setup
#run mkdir -p /opt/algorithm/nnunet_raw_data_base/nnunet_raw_data/task001_tcia/imagests /opt/algorithm/nnunet_raw_data_base/nnunet_raw_data/task001_tcia/result 
run mkdir -p "/opt/algorithm/nnUNet_raw_data_base/nnunet_raw_data/Task501_autoPET/imagesTs" "/opt/algorithm/nnUNet_raw_data_base/nnunet_raw_data/Task501_autoPET/result"
run ls /opt/algorithm
#'/opt/algorithm/nnUNet_raw_data_base/nnunet_raw_data/Task501_autoPET/imagesTs'
env nnUNet_raw_data_base="/opt/algorithm/nnUNet_raw_data_base" 
env RESULTS_FOLDER="/opt/algorithm/weights"
env MKL_SERVICE_FORCE_INTEL=1

entrypoint python -m infer $0 $@
