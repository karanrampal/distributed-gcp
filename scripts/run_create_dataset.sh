#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Navigate to src directory
cd "$(dirname "$0")"
cd ../src

# Set variables
runner="DataflowRunner"
proj="smle-attribution-d237"
region="europe-west4"
tmp_loc="gs://smle-temp-bucket/dataflow/tmp"
subname="smle-vpc-subnet"
subnet="https://www.googleapis.com/compute/v1/projects/${proj}/regions/${region}/subnetworks/${subname}"

echo "Running script"
python create_dataset.py --runner $runner --project $proj --region $region --temp_location $tmp_loc --subnetwork $subnet