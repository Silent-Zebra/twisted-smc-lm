#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Error: Please provide the training command"
    exit 1
fi

# Store the full command
COMMAND="$*"


# Extract multiple parameters in one go

PARAMS=$(echo "$COMMAND" | awk '
{
    # Initialize empty variables
    len = lr = lrp = beta = seed = rm = ntwist = npolicy = twist_up = policy_up = ""
    model = twist_learn = rl_loss = alpha = baseline = threshold = neg_train = adaptive_baseline = separate_twist = backprop_through = ""
    has_alpha = has_baseline = 0

    # Scan through all matches in the string
    for(i=1; i<=NF; i++) {
        if($i == "--output_len" ) len = $(i+1)
        if($i == "--lr_twist" ) lr = $(i+1)
        if($i == "--lr_p" ) lrp = $(i+1)
        if($i == "--beta_temp" ) beta = $(i+1)
        if($i == "--seed" ) seed = $(i+1)
        if($i == "--rm_type" ) rm = $(i+1)
        if($i == "--n_twist" ) ntwist = $(i+1)
        if($i == "--n_policy_samples" ) npolicy = $(i+1)
        if($i == "--twist_updates_per_epoch" ) twist_up = $(i+1)
        if($i == "--policy_updates_per_epoch" ) policy_up = $(i+1)
        if($i == "--hface_model_type" ) model = $(i+1)
        if($i == "--twist_learn_type" ) twist_learn = $(i+1)
        if($i == "--rl_loss_type" ) rl_loss = $(i+1)
        
        # Optional parameters
        if($i == "--alpha_adv" ) {
            alpha = "_alpha" $(i+1)
            has_alpha = 1
        }
        if($i == "--use_hardcoded_baseline") {
            has_baseline = 1
        }
        if($i == "--hardcoded_baseline"  && has_baseline) {
            baseline = "_baseline" $(i+1)
        }
     
        if($i == "--negative_training_threshold" ) {
            neg_train = "_threhsold" $(i+1)
        }
        if($i == "--adaptive_baseline_percentile" ) {
            adaptive_baseline = "_adaptive" $(i+1)
        }
	if($i == "--separate_hface_twist_model" ) {
            separate_twist = "_separatetwist"
        }
	if($i == "--backprop_twist_through_backbone" ) {
            backprop_through = "_backpropthrough"
        }
        if(match($i, /--threshold=(-?[0-9.]+)/)  ) {
            threshold = substr($i, RSTART+12, RLENGTH-12)
        }
    }
    # Print with a special delimiter (|) that wont appear in the values
    if(len != "" && lr != "" && lrp != "" && beta != "" && seed != "" && rm != "" &&
       ntwist != "" && npolicy != "" && twist_up != "" && policy_up != "" &&
       model != "" && twist_learn != "" && rl_loss != "")
        print len "|" lr "|" lrp "|" beta "|" seed "|" rm "|" ntwist "|" npolicy "|" \
              twist_up "|" policy_up "|" model "|" twist_learn "|" rl_loss "|" \
              alpha "|" baseline "|" threshold "|" neg_train "|" adaptive_baseline "|" separate_twist "|" backprop_through
}')


# Read using the special delimiter
IFS='|' read OUTPUT_LEN LR_TWIST LR_P BETA_TEMP SEED RM_TYPE N_TWIST N_POLICY TWIST_UPDATES \
     POLICY_UPDATES MODEL TWIST_LEARN_TYPE RL_LOSS_TYPE ALPHA_ADV BASELINE THRESHOLD NEG_TRAIN ADAPTIVE_BASELINE SEPARATE_TWIST BACKPROP_THROUGH <<< "$PARAMS"


# Check if required parameters are empty
if [ -z "$OUTPUT_LEN" ] || [ -z "$LR_TWIST" ] || [ -z "$LR_P" ] || [ -z "$BETA_TEMP" ] || [ -z "$SEED" ] || \
   [ -z "$RM_TYPE" ] || [ -z "$N_TWIST" ] || [ -z "$N_POLICY" ] || [ -z "$TWIST_UPDATES" ] || \
   [ -z "$POLICY_UPDATES" ] || [ -z "$MODEL" ] || [ -z "$TWIST_LEARN_TYPE" ] || [ -z "$RL_LOSS_TYPE" ]; then
    echo "Error: Missing required parameters"
    exit 1
fi

# Get current date in required format
CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)

# Generate output filename
PATTERN="${CURRENT_DATE}_${RM_TYPE}${THRESHOLD}_${MODEL}_beta${BETA_TEMP}_len${OUTPUT_LEN}_batch${N_TWIST}_${N_POLICY}_${TWIST_UPDATES}${TWIST_LEARN_TYPE}_${LR_TWIST}_${POLICY_UPDATES}${RL_LOSS_TYPE}${NEG_TRAIN}_${LR_P}${ALPHA_ADV}${BASELINE}${ADAPTIVE_BASELINE}${SEPARATE_TWIST}${BACKPROP_THROUGH}"

echo $SEPARATE_TWIST
echo $BACKPROP_THROUGH
exit 1

SBATCH_FILE="sbatch_${PATTERN}"
OUTPUT_FILE="result_${PATTERN}_s1.txt"



# Create the sbatch file
cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --partition=a40
#SBATCH --qos=m3
#SBATCH --export=ALL
#SBATCH --output=$OUTPUT_FILE
#SBATCH --gres=gpu:1
. activate smc-lm
export LD_LIBRARY_PATH="/pkgs/cudnn-11.x-v8.9.6/lib64:/pkgs/cuda-11.8/lib64"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/pkgs/cuda-11.8"
export PATH="/pkgs/cuda-11.8/bin:\$PATH"
XLA_PYTHON_CLIENT_MEM_FRACTION=.5 $COMMAND
EOL

# Make the sbatch file executable
chmod +x "$SBATCH_FILE"

echo "Created sbatch file: $SBATCH_FILE"
echo "Output will be written to: $OUTPUT_FILE"
