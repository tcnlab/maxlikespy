touch results/cell_fits.txt
touch results/model_comparisons.txt

for i in `seq 0 10`; do

	qsub -pe omp 1 -l h_rt=140:00:00 -V ./run_one_cell.sh $i
	pids[$i]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done

qsub -pe omp 1 -l h_rt=140:00:00 -V ./stich_output.py 0 10