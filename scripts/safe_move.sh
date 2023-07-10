#!/bin/bash

########################################################################################
# Script Name: Safe File Mover with Checksum
# Version: v3
#
# Description: This script safely moves files from one directory to another. It calculates
# checksums of the files before and after moving, ensuring data integrity. The script preserves
# the directory structure in the destination directory with or without permissions or authorities.
# Filenames with spaces and newlines are handled correctly.
#
# Backend : Rsync and CP
# Supported OS: This version of the script is compatible with Linux, macOS.
# Caveat: This script does not provide chunking, and native (cp) parallel execution
#
# How to use:
# 1. Download the script and make it executable with the following command:
#    chmod +x safe_move.sh
# 2. Run the script with the following arguments:
#    ./safe_move.sh /path/to/sourceDirectory /path/to/destinationDirectory
# 3. Optional: Add --rsync to use rsync instead of cp
# 4. Optional: Add --no-preserve to not preserve permissions
#
# Author: Alan Synn (alan@alansynn.com)
# Date: June 5th, 2018
# Last Edited: Aug 10th, 2021
########################################################################################

####################################### UTILITIES ######################################

# Function to check if command exists
command_exists () {
    type "$1" &> /dev/null ;
}

######################################## SETUP #########################################

# Determine the platform and appropriate checksum command
platform=$(uname)
if [[ $platform == 'Linux' ]]; then
    if command_exists sha256sum ; then
        checksum_cmd='sha256sum'
    else
        echo -e "\033[31msha256sum command not found. Please install it before running the script.\033[0m"
        exit 1
    fi
elif [[ $platform == 'Darwin' ]]; then
    if command_exists sha3sum ; then
        checksum_cmd='sha3sum'
    else
        echo -e "\033[31mshasum command not found. Please install it before running the script.\033[0m"
        exit 1
    fi
else
    echo -e "\033[31mUnsupported platform: $platform\033[0m"
    exit 1
fi

######################################## PARSER ########################################

# Check if the right number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./safe_move.sh /path/to/sourceDirectory /path/to/destinationDirectory"
    echo "Optional: --rsync, --no-preserve"
    exit 1
fi

src="$1"
dst="$2"

# Check if rsync is requested
use_rsync=0
if [[ "$3" == "--rsync" || "$4" == "--rsync" ]]; then
    use_rsync=1
fi

# Check if permissions should not be preserved
preserve_permissions=1
if [[ "$3" == "--no-preserve" || "$4" == "--no-preserve" ]]; then
    preserve_permissions=0
fi

######################################### Main #########################################

# Set the IFS variable to handle filenames with spaces
IFS=$'\n'

# Create the destination directory if it doesn't exist
mkdir -p "$(dirname "$dst")"

find "$src" -type f -print0 | while IFS= read -r -d $'\0' srcfile
do
    # Replace the initial part of the file path with the destination directory
    dstfile="${srcfile/$src/$dst}"

    # Create the destination directory if it doesn't exist
    mkdir -p "$(dirname "$dstfile")"

    # Copy the source file to the destination
    if [[ $use_rsync -eq 1 ]]; then
        rsync -avhW --compress-level=0 --checksum --remove-source-files "$srcfile" "$dstfile" | sed '/\/$/d'
    else
        if [[ $preserve_permissions -eq 1 ]]; then
        cp -a "$srcfile" "$dstfile"
        else
        cp "$srcfile" "$dstfile"
        fi
    fi

    # Check the SHA256 checksum of both files
    src_checksum=$($checksum_cmd "$srcfile" | awk '{ print $1 }')
    dst_checksum=$($checksum_cmd "$dstfile" | awk '{ print $1 }')

    # If the checksums match, delete the source file
    if [ "$src_checksum" == "$dst_checksum" ]
    then
        rm "$srcfile"
        echo -e "\033[32m[Log] Moved $srcfile to $dstfile\033[0m"

        # Find and delete empty directories in the source directory
        find "$src" -type d -empty -delete
    else
        echo -e "\033[31m[Error] Checksums do not match for $srcfile\033[0m"
        exit 1
    fi
done
