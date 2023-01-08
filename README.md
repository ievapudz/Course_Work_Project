# Protein classification prediction using sequence representations from protein language models - course work project

This repository contains files that were used to retrieve results presented
in the thesis of this work.

## Preparing the data set

Splitting of FASTA files was done using
[`fasta-splitter`](http://kirill-kryukov.com/study/tools/fasta-splitter/)
program:

```
./fasta-splitter.pl --n-parts 60 --out-dir data/003/FASTA/training_v2_ndup_filtEmb/ \
    --nopad --part-num-prefix '_' data/003/FASTA/training_v2_ndup_filtEmb.fasta

./fasta-splitter.pl --n-parts 7 --out-dir data/003/FASTA/validation_v2_ndup_filtEmb/ \
    --nopad --part-num-prefix '_' data/003/FASTA/validation_v2_ndup_filtEmb.fasta

./fasta-splitter.pl --n-parts 7 --out-dir data/003/FASTA/testing_v2_ndup_filtEmb/ \
    --nopad --part-num-prefix '_' data/003/FASTA/testing_v2_ndup_filtEmb.fasta
```

Batch generation of embeddings:

```
mkdir -p data/003/slurm/

sbatch \
    --export=ALL,DATA_DIR='data/003/',JOB_NAME='training_v2_ndup_filtEmb',REPRESENTATIONS='per_res mean quantile' \
    --job-name='training_v2_ndup_filtEmb' \
    --output=data/003/slurm/training_v2_ndup_filtEmb_%a.out --array=1-60 scripts/ESM-1b/embeddings.sh

sbatch \
    --export=ALL,DATA_DIR='data/003/',JOB_NAME='validation_v2_ndup_filtEmb',REPRESENTATIONS='per_res mean quantile' \
    --job-name='validation_v2_ndup_filtEmb' \
    --output=data/003/slurm/validation_v2_ndup_filtEmb_%a.out --array=1-7 scripts/ESM-1b/embeddings.sh

sbatch \
    --export=ALL,DATA_DIR='data/003/',JOB_NAME='testing_v2_ndup_filtEmb',REPRESENTATIONS='per_res mean quantile' \
    --job-name='testing_v2_ndup_filtEmb' \
    --output=data/003/slurm/testing_v2_ndup_filtEmb_%a.out --array=1-7 scripts/ESM-1b/embeddings.sh

sbatch --export=ALL,DATA_DIR='data/003/',JOB_NAME='training_v2_ndup_filtEmb' --array=1-60 \
    --output=data/003/slurm/training_v2_PT_ndup_filtEmb_%a.out scripts/ProtTrans/embeddings.sh

sbatch --export=ALL,DATA_DIR='data/003/',JOB_NAME='validation_v2_ndup_filtEmb' --array=1-7 \
    --output=data/003/slurm/validation_v2_PT_ndup_filtEmb_%a.out scripts/ProtTrans/embeddings.sh

sbatch --export=ALL,DATA_DIR='data/003/',JOB_NAME='testing_v2_ndup_filtEmb' --array=1-7 \
    --output=data/003/slurm/testing_v2_PT_ndup_filtEmb_%a.out scripts/ProtTrans/embeddings.sh

```

Saving embeddings:

```
./scripts/ESM-1b/save_embeddings.py \
    -f data/003/FASTA/data/003/FASTA/training_v2_ndup_filtEmb.fasta \
    --pt-dir data/003/ESM-1b/ --representations 'mean' \
    --keyword 'validate' -t data/003/TSV/training_v2_ESM_ndup_filtEmb.tsv 

time srun ./scripts/ProtTrans/save_embeddings.py \
    --fasta data/003/FASTA/training_v2_ndup_filtEmb.fasta --pt-dir data/003/ProtTans \
    --keyword 'train' --representations 'mean' \
    --tsv data/003/TSV/training_v2_PT_ndup_filtEmb.tsv
```

converting TSV to NPZ:

```
./scripts/TSV_to_NPZ.py \
    -t data/003/TSV/training_v2_ESM_ndup_filtEmb.tsv \
    -k 'train' -n data/003/NPZ/training_v2_ESM_ndup_filtEmb.npz

./scripts/TSV_to_NPZ.py \
    -t data/003/TSV/training_v2_PT_ndup_filtEmb.tsv \
    -k 'train' -n data/003/NPZ/training_v2_PT_ndup_filtEmb.npz
```

To get joined embeddings:

```
time srun ./scripts/ProtTrans/join_embeddings.py \
    --npz-1 data/003/NPZ/training_v2_ESM_ndup_filtEmb.npz \
    --npz-2 data/003/NPZ/training_v2_PT_ndup_filtEmb.npz \
    -k 'train' \
    --npz-out data/003/NPZ/training_v2_ndup_filtEmb_joined.npz \
    --tsv-out data/003/TSV/training_v2_ndup_filtEmb_joined.tsv
```

To get per-residue representations percentiles (for this example, octiles):

```
./scripts/pick_percentiles.py -f data/003/FASTA/training_v2_ndup_filtEmb.fasta
	--pt-dir data/003/ESM-1b/ --keyword 'train' \
	--percentiles '0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1' \
	-t  data/003/TSV/training_v2_ESM_ndup_filtEmb_octiles.tsv
```

## Correlation analysis

The small set for correlation analysis was created using the first four parts
of split validation set (003). Sequences were filtered to keep only those that 
have embeddings generated.

```
./scripts/ESM-1b/save_embeddings.py \
    -f data/003/FASTA/validation_small_set_2_filtered.fasta \
    --pt-dir data/003/ESM-1b/ --representations 'mean' \
    --keyword 'validate' -t data/003/TSV/validation_small_set_2_filtered.tsv \
    -n data/003/NPZ/validation_small_set_2_filtered.npz

time srun ./scripts/ProtTrans/save_embeddings.py \
    --fasta data/003/FASTA/validation_small_set_2_filtered.fasta --pt-dir data/003/ProtTans \
    --keyword 'validate' --representations 'mean' \
    --tsv data/003/TSV/validation_small_set_2_PT_filtered.tsv

./scripts/ProtTrans/TSV_to_NPZ.py \
    -t data/003/TSV/validation_small_set_2_ESM_filtered.tsv \
    -k 'validate' -n data/003/NPZ/validation_small_set_2_ESM_filtered.npz

./scripts/ProtTrans/TSV_to_NPZ.py \
    -t data/003/TSV/validation_small_set_2_PT_filtered.tsv \
    -k 'validate' -n data/003/NPZ/validation_small_set_2_PT_filtered.npz

time srun ./scripts/ProtTrans/join_embeddings.py \
    --npz-1 data/003/NPZ/validation_small_set_2_ESM_filtered.npz \
    --npz-2 data/003/NPZ/validation_small_set_2_PT_filtered.npz \
    -k 'validate' \
    --npz-out data/003/NPZ/validation_small_set_2_joined.npz \
    --tsv-out data/003/TSV/validation_small_set_2_joined.tsv
```

Matrices with correlation coefficients were created by executing 
the following commands:

```
./scripts/correlation.py --npz validation_small_set_2_joined.npz \
	-k 'validate' -b 1280 \
	--matrix-out data/validation_small_set_2_joined_matrix.txt
```

```
./scripts/correlation.py --npz validation_small_set_2_joined.npz \
    -k 'validate' -b 1280 \
    --matrix-out data/validation_small_set_2_joined_PC_95_matrix.txt
```

Plotting for correlation between embeddings' components analysis:

```
./scripts/correlation_stats.py -i data/validation_small_set_2_joined_matrix.txt \
	-t 0.5 --annot-threshold 10 \
	--high-corr thesis/figures/validation_small_set_2_joined_correlation_high_corr.png \
	--title "Numbers of absolute correlation coefficients with high values"
```

```
./scripts/correlation_stats.py -i data/validation_small_set_2_joined_matrix.txt \
	--hist thesis/figures/validation_small_set_2_joined_correlation_hist.png \
	--title "Numbers of correlation coefficients"
```

```
./scripts/correlation_stats.py -i data/validation_small_set_2_joined_matrix.txt \
    --max thesis/figures/validation_small_set_2_joined_correlation_max.png \
    --title "Plot of absolute maximum correlation coefficients"
```

```
./scripts/correlation_stats.py -i data/validation_small_set_2_joined_matrix.txt \
    --mean thesis/figures/validation_small_set_2_joined_correlation_mean.png \
    --title "Plot of absolute mean correlation coefficients"
```

```
./scripts/correlation_stats.py -i data/validation_small_set_2_joined_PC_95_matrix.txt \
    --max thesis/figures/validation_small_set_2_joined_PC_95_correlation_max.png \
    --title "Plot of absolute maximum correlation coefficients"
```

```
./scripts/correlation_stats.py -i data/validation_small_set_2_joined_PC_95_matrix.txt \
    --mean thesis/figures/validation_small_set_2_joined_PC_95_correlation_mean.png \
    --title "Plot of absolute mean correlation coefficients"
```

## Model training and evaluation workflow

An example command to run in order to execute 
training, validation, and testing processes for an SLP:

```
mkdir -p results/SLP/003/ESM_ndup_filtEmb/ROC/

srun ./scripts/003/classifier.py \
    --npz-train data/003/NPZ/training_v2_ESM_ndup_filtEmb.npz \
    --npz-validate data/003/NPZ/validation_v2_ESM_ndup_filtEmb.npz \
    --npz-test data/003/NPZ/testing_v2_ESM_ndup_filtEmb.npz -a SLP \
    -m results/SLP/003/ESM_ndup_filtEmb/ -i 1280 \
    -p results/SLP/003/ESM_ndup_filtEmb/ -r results/SLP/003/ESM_ndup_filtEmb/ROC/
```

To train models with hidden layers:

```
mkdir -p results/MLP_C2H2_h512-256/003/PT_ndup_filtEmb/ROC/

srun ./scripts/003/classifier.py \
    --npz-train data/003/NPZ/training_v2_PT_ndup_filtEmb.npz \
    --npz-validate data/003/NPZ/validation_v2_PT_ndup_filtEmb.npz \
    --npz-test data/003/NPZ/testing_v2_PT_ndup_filtEmb.npz -a MLP \
    -m results/MLP_C2H2_h512-25/003/PT_ndup_filtEmb/ -i 1024 \
	--hidden '512 256' \
    -p results/MLP_C2H2_h512-25/003/PT_ndup_filtEmb/ -r results/MLP_C2H2_h512-25/003/PT_ndup_filtEmb/ROC/
```

Testing scores can be checked by running and taking into account
the last record of the output:

```
grep '#' results/SLP/003/ESM_ndup/predictions_l1e-04_b24_e5.tsv | less
```
