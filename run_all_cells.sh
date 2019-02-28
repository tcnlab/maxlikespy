#!/bin/bash -l
#$ -l h_rt=140:00:00
#$ -P ecog-eeg
#$ -pe omp 1


module load python3

mkdir results
mkdir results/figs
touch results/cell_fits.txt
touch results/model_comparisons.txt
touch results/log_likelihoods.txt

first = $1
last = $2
numJobs=$((last-first))     # Count the jobs
myJobIDs=""                            # Initialize an empty list of job IDs
for i in `seq $1 $2`; do
    jobID_full=$(qsub ./run_one_cell.sh $i)
    # jobID_full will look like "12345.machinename", so use sed
    # to get just the numbers
    jobID=$(echo "$jobID_full" | sed -e 's|\([0-9]*\).*|\1|')
    myJobIDs="$myJobIDs $jobID"        # Add this job ID to our list
done

numDone=0                              # Initialize so that loop starts
while [ $numDone -lt $numJobs ]; do    # Less-than operator
    numDone=0                          # Zero since we will re-count each time
    for jobID in $myJobIDs; do         # Loop through each job ID

        # The following if-statement ONLY works if qstat won't return
        # the string ' C ' (a C surrounded by two spaces) in any
        # situation besides a completed job.  I.e. if your username
        # or jobname is 'C' then this won't work!
        # Could add a check for error (grep -q ' E ') too if desired
        if qstat $jobID | grep -q ' C ' 
        then
            (( numDone++ ))
        else
            echo $numDone jobs completed out of $numJobs
            sleep 1
        fi
    done
done

#qsub -pe omp 1 -l h_rt=140:00:00 -V python3 stitch_output.py 0 10
python3 stitch_output.py $1 $2