export DATA_DIR="/home/server08/hdd1/yoonjeon_workspace"

# Check if DATA_DIR exists. If it does, create a symlink to it as "./data".
if [ -d "${DATA_DIR}" ]; then
    ln -s "${DATA_DIR}" ./data
else
    # Otherwise, use the alternate DATA_DIR and create a symlink to that.
    export DATA_DIR="/data/yoonjeon_workspace"
    ln -s "${DATA_DIR}" ./data
fi

# Set COCO_DIR as a subdirectory of DATA_DIR and create it if it doesn't exist.
export COCO_DIR="${DATA_DIR}/coco/"
mkdir -p "${COCO_DIR}"

for split in train val test; do
    if [ ! -d "${COCO_DIR}/${split}2017" ]; then
        echo "Downloading COCO dataset..."
        wget "http://images.cocodataset.org/zips/${split}2017.zip"
        unzip "${split}2017.zip" -d "${COCO_DIR}"
        rm "${split}2017.zip"
    fi
done


wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip

export AOKVQA_DIR=${DATA_DIR}/aokvqa/
mkdir -p ${AOKVQA_DIR}

curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}