#!/bin/bash

# It is a submission script to run embeddings'
# generation program.

set -ue

FASTA_PREFIX=${DATA_DIR}/FASTA/${JOB_NAME}_

[ ! -d "${DATA_DIR}/ProtTrans/${JOB_NAME}" ] && mkdir -p ${DATA_DIR}/ProtTrans/${JOB_NAME}/

echo "Job started at $(date)"

./scripts/ProtTrans/generate_embeddings.py -f \
	${FASTA_PREFIX}${SLURM_ARRAY_TASK_ID}.fasta -d \
	${DATA_DIR}/ProtTrans/${JOB_NAME}/ -r "${REPRESENTATIONS}" \
	--model ${MODEL_DIR}
	
echo "Job end at $(date)"

