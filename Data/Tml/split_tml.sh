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
        echo "Error: The sum of percentages must equal 100."
        exit 1
    fi
fi

mkdir -p train val test

EVENTS=$(ls "$DATASET_DIR" | grep -o 'event[0-9]\{9\}' | sort -u)

# Check events
if [ -z "$EVENTS" ]; then
    echo "Error: No events found in the dataset directory."
    exit 1
fi

EVENT_ARRAY=($EVENTS)
total_events=${#EVENT_ARRAY[@]}

# events for each split
train_count=$((total_events * TRAIN_PERCENT / 100))
val_count=$((total_events * VAL_PERCENT / 100))
test_count=$((total_events - train_count - val_count))

total_success=0
total_files=$((total_events * 4))

prev_dest_dir=""

echo "Starting the copy process..."
echo "Total events to process: ${#EVENT_ARRAY[@]}"
echo "Total expected files: $total_files"
echo ""

for i in "${!EVENT_ARRAY[@]}"; do
    event=${EVENT_ARRAY[$i]}
    
    if [ "$i" -lt "$train_count" ]; then
        dest_dir="train"
    elif [ "$i" -lt "$((train_count + val_count))" ]; then
        dest_dir="val"
    else
        dest_dir="test"
    fi

    for file_type in particles truth hits cells; do
        src_file="$DATASET_DIR/$event-$file_type.csv"
        dest_file="$dest_dir/$event-$file_type.csv"
        
        if [ -f "$src_file" ]; then
            cp "$src_file" "$dest_file" && ((total_success++))
           
            printf "[%3d%%] \t Destination: %s \t Copied: %s-%s.csv \r" $((total_success * 100 / total_files)) "$dest_dir" "$event" "$file_type"
        else
            echo "Warning: Source file $src_file does not exist."
        fi
    done
done

echo ""
echo ""

echo "============================"
echo "         Summary            "
echo "============================"
echo "Total Copies: $total_success / $total_files" 
echo "============================"
