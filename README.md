# Protein classification prediction using sequence representations from protein language models - course work project

This repository contains files that were used to retrieve results presented
in the thesis of this work.

## Correlation analysis

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

Plotting:

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

## To-do
- [] Create TSV or NPZ files for 'validation_small_set_2_joined'
- [] Describe the general flow to get the required data set for reproducibility and run the training and evaluation process
