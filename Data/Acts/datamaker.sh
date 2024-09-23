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





mkdir -p train val test







EVENTS=$(ls "$DATASET_DIR" | grep -o 'event[0-9]\{9\}' | sort -u)

# Check events
if [ -z "$EVENTS" ]; then
    echo "Error: No events found in the dataset directory."
    exit 1
fi


EVENT_ARRAY=($EVENTS)
total_events=${#EVENT_ARRAY[@]}

train_count=$((total_events * TRAIN_PERCENT / 100))
val_count=$((total_events * VAL_PERCENT / 100))  # split ratios
test_count=$((total_events - train_count - val_count))



total_success=0
total_files=$((total_events * 4)) 

for i in "${!EVENT_ARRAY[@]}"; do
    event=${EVENT_ARRAY[$i]}
    
    
    if [ "$i" -lt "$train_count" ]; then
        dest_dir="train"
        echo "copying to train..."
    elif [ "$i" -lt "$((train_count + val_count))" ]; then
        dest_dir="val"
        echo "copying to val..."
    else
        dest_dir="test"
        echo "copying to test..."
    fi

    # Copy 
    for file_type in parameters particles spacepoint tracks; do
        src_file="$DATASET_DIR/$event-$file_type.csv"
        dest_file="$dest_dir/"

        if cp "$src_file" "$dest_file"; then
            ((total_success++))
        fi
    done
done


echo "Copying summary:"

echo "Total: $total_success files out of $total_files"

echo "data splited ...."
