#!/bin/bash

# Check if filename is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

filename="$1"

# Check if file exists
if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' not found"
    exit 1
fi


# Extract the base output filename from the SBATCH directive and remove _s1.txt suffix
base_output_file=$(grep "^#SBATCH --output=" "$filename" | cut -d'=' -f2 | sed 's/_s[0-9]\+\.txt$//')

echo = \[

# Process files for seeds 1-5
for seed in `seq 1 5`; do
    
    # Generate output filename for current seed
    current_output_file="${base_output_file}_s${seed}.txt"
    
    # echo "\n# Output file (seed ${seed}): $current_output_file"
    
    # Check if output file exists
    if [ ! -f "$current_output_file" ]; then
        echo "Warning: Output file '$current_output_file' not found"
        continue
    fi
    
    # Find the last occurrence of "Information collected over time" and extract the dictionary before it
    # echo "# Last information collection:"
    tac "$current_output_file" | \
        awk '{ prev = curr; curr = $0 }
             /Information collected over time:/ { printf "%s,\n", prev; exit } ' | \
        tac

done
    
echo \]

# Print filename
echo "# $filename"

# Find and print the Python command
# Look for a line starting with "XLA_PYTHON_CLIENT_MEM_FRACTION=" or just "python"
# and extract everything after "python"
grep -E "^(XLA_PYTHON_CLIENT_MEM_FRACTION=.*)?python" "$filename" | \
    sed -E 's/^XLA_PYTHON_CLIENT_MEM_FRACTION=.*python/# python/' | \
    sed -E 's/^python/# python/'
