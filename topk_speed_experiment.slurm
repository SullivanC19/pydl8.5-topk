for SLURM_ARRAY_TASK_ID in {0..95}
do

DEPTHS=(1 2 3 4 5 6 7 8)
KS=(1 2 4 8 12 16)
DATASETS=("connect-4" "artifical-characters")

depth_idx=$((1 + (SLURM_ARRAY_TASK_ID % 8)))
k_idx=$((1 + ((SLURM_ARRAY_TASK_ID / 8) % 6)))
dataset_idx=$((1 + ((SLURM_ARRAY_TASK_ID / 48) % 2)))

depth=${DEPTHS[depth_idx]}
k=${KS[k_idx]}
dataset=${DATASETS[dataset_idx]}

echo $depth
echo $k
echo $dataset

python run_topk.py -d ${dataset} -k ${k} -m ${depth} -to > topk/results/speed/${dataset}_k=${k}_depth=${depth}.out

done