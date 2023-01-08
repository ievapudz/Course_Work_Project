#!/bin/bash

# It is a submission script to run embeddings'
# generation program.

set -ue

FASTA_PREFIX=${DATA_DIR}/FASTA/${JOB_NAME}_

[ ! -d "${DATA_DIR}/ESM-1b/" ] && mkdir ${DATA_DIR}/ESM-1b/

[ ! -d "${DATA_DIR}/ESM-1b/${JOB_NAME}/" ] && mkdir ${DATA_DIR}/ESM-1b/${JOB_NAME}/

echo "Job started at $(date)"

./scripts/ESM-1b/generate_embeddings.py \
	-f ${FASTA_PREFIX}${SLURM_ARRAY_TASK_ID}.fasta --no_filter \
	-r "${REPRESENTATIONS}" -e ${DATA_DIR}/ESM-1b/${JOB_NAME}/ \
	
echo "Job end at $(date)"

