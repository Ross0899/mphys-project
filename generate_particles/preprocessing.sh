#!/bin/bash

# Script to automate the generation and preprocessing of the synthetically 
# generated particle images.

# Usage ./preprocess.sh [number of 512x512 images to produce]

set -e

# Parse user command line argument
re='^[0-9]+$'
if [ "$1" == "" ] || [ $# -lt 1 ]; then

    echo "error: Please enter the number of images to be generated."
    echo "Usage: ./preprocess.sh [number of 512x512 images to produce]"
    exit 1
fi


if ! [[ $1 =~ $re ]] ; then
   echo "error: Command line argument is not a number." >&2
   exit 1
fi

# Run pre-processing 
echo "#################################"
echo "Pre-processing CNN Training Data "
echo "#################################"

# Remove exisiting pre-processed images
if [ -d "./data/synthetic/particles" ]; then
    rm -rf ./data/synthetic/particles/* ; fi

if [ -d "./data/synthetic/masks" ]; then
        rm -rf ./data/synthetic/masks/* ; fi

if [ -d "./data/overlays/" ]; then
    rm -rf ./data/overlays/* ; fi

if [ -d "./data/divided/particles" ]; then
    rm -rf ./data/divided/particles/* ; fi

if [ -d "./data/divided/masks" ]; then
    rm -rf ./data/divided/masks/* ; fi 

if [ -d "./data/augmented/particles" ]; then
    rm -rf ./data/augmented/particles/* ; fi

if [ -d "./data/augmented/masks" ]; then
    rm -rf ./data/augmented/masks/* ; fi

# Run Python scripts
echo "Generating..."
python3 v1_generating_particles.py $1
echo ""
echo "Overlaying..."
#python3 overlay_mask_image.py
echo "Dividing..."
python3 divide_images_to_patches.py
echo "Augmenting..."
python3 augment_images.py

# Output to user
echo ""
echo "##################################"
echo "Pre-processing successful"

NUM_PARTICLE_FILES=$(ls ./data/augmented/particles | wc -l)
NUM_MASK_FILES=$(ls ./data/augmented/masks | wc -l)

echo "${NUM_PARTICLE_FILES}" training images produced.
echo "${NUM_MASK_FILES}" training masks produced.
