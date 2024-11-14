#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset_dir> <train_percentage> <val_percentage> <test_percentage>"
    exit 1
else
    DATASET_DIR=$1
    TRAIN_PERCENT=$2
    VAL_PERCENT=$3
    TEST_PERCENT=$4

    SUM=$((TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT))
    if [ "$SUM" -ne 100 ]; then
        echo "Error: use Percentage ratios"
        exit 1
    fi
fi

# Create directories
mkdir -p train val test

# Extract events
EVENTS=$(ls "$DATASET_DIR" | grep -o 'event[0-9]\{9\}' | sort -u)
if [ -z "$EVENTS" ]; then
    echo "Error: dataset directory empty"
    exit 1
fi

EVENT_ARRAY=($EVENTS)
total_events=${#EVENT_ARRAY[@]}

train_count=$((total_events * TRAIN_PERCENT / 100))
val_count=$((total_events * VAL_PERCENT / 100))
test_count=$((total_events - train_count - val_count))

total_success=0
total_files=$((total_events * 4)) # 4 files per event



echo "Starting the spliting process..."
echo "Total events to process: ${#EVENT_ARRAY[@]}"
echo "Total expected files: $total_files"
echo ""


for i in "${!EVENT_ARRAY[@]}"; do
    event=${EVENT_ARRAY[$i]}

    # Destination directory
    if [ "$i" -lt "$train_count" ]; then
        dest_dir="train"
    elif [ "$i" -lt "$((train_count + val_count))" ]; then
        dest_dir="val"
    else
        dest_dir="test"
    fi

    
    for file_type in parameters particles spacepoint tracks; do
        src_file="$DATASET_DIR/$event-$file_type.csv"
        dest_file="$dest_dir/"

        if cp "$src_file" "$dest_file"; then
            ((total_success++))
        fi

        # progress
        progress=$((total_success * 100 / total_files))
        echo -ne "\r[$progress%]  Destination: $dest_dir  Copied: $event-$file_type.csv"
    done
done


# Summary 
echo ""
echo "============================"
printf "%-20s | %-10s\n" "Category" "Count"
echo "============================"
printf "%-20s | %-10d\n" "Total Files Copied" "$total_success"
echo "============================"
