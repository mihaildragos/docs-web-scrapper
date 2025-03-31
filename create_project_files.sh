#!/bin/bash

# Create or clear the output file
echo "# Project Files" >project_files.md

# Set maximum file size (in bytes) - 1MB
MAX_SIZE=$((1024 * 1024))

# Count total files for progress tracking
total_files=$(git ls-files | wc -l)
current_file=0

# Make a list of files to include
files_to_include=(
)

files_to_exclude=(
    "create_project_files.sh"
)



# Get all tracked files using git ls-files
git ls-files | while read -r file; do
    # Update progress counter
    ((current_file++))
    echo -ne "Processing file $current_file of $total_files\r"
    echo -ne "\n## $file\n"

    # Skip files that start with ., package-lock.json, or are in build/dist folders
    if [[ ! "$file" =~ ^\. ]] &&
        [[ ! "$file" =~ package-lock\.json$ ]] &&
        [[ ! "$file" =~ project_files\.md$ ]] &&
        #  files with these extensions
        [[ ! "$file" =~ \.(scss)$ ]] &&
        # Don't include secret files like .env files, even the ones in sub-folders or named like .env.*
        [[ ! "$file" =~ \.(env|env.*|gitignore|backend/.env)$ ]] &&
        # Don't include readme files in any sub-folders (i.e. backend/README.md)
        [[ ! "$file" =~ \/(README\.md)$ ]] &&
        # Don't include files in the build/dist/node_modules folders
        [[ ! "$file" =~ ^(build|dist|node_modules|backend/node_modules|\.terraform|venv|output|session_data)/ ]] &&
        [[ ! "$file" =~ ^(backend/__tests__)/ ]] &&
        # # Only include files in the files_to_include array
        # [[ " ${files_to_include[@]} " =~ " $file " ]]; then
        # Don't include files in the files_to_exclude array
        [[ ! " ${files_to_exclude[*]} " =~ $file ]]; then
        

        # Check if file is binary
        if file "$file" | grep -q "text"; then
            # Check file size
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)

            if [ "$size" -lt "$MAX_SIZE" ]; then
                # Add file header with path
                echo -e "\n## $file\n" >>project_files.md
                # Add file contents in a code block with appropriate language
                echo "\`\`\`${file##*.}" >>project_files.md
                cat "$file" >>project_files.md
                echo "\`\`\`" >>project_files.md
            else
                echo -e "\n## $file\n" >>project_files.md
                echo "*File too large to display ($(($size / 1024)) KB)*" >>project_files.md
            fi
        else
            echo -e "\n## $file\n" >>project_files.md
            echo "*Binary file - contents not shown*" >>project_files.md
        fi
    fi
done

# Clear progress line and show completion message
echo -e "\nDone! Generated project_files.md"
