#!/usr/bin/env python3
"""
Split graded JSON files into separate case files
"""

import os
import json
import sys
from pathlib import Path

def split_json_file(json_path, output_name):
    """
    Split a graded JSON file into separate case files.
    Creates a subfolder with the specified name and saves each case as a separate JSON file.
    """
    # Read the input JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return

    # Create output directory
    output_dir = Path('graded') / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")

    # Get metadata if it exists
    metadata = data.get('metadata', {})
    cases = data.get('cases', data)  # If no 'cases' key, assume entire dict is cases

    # Split each case into its own file
    for case_id, case_data in cases.items():
        # Create output file path
        case_file = output_dir / f"{case_id}.json"
        
        # Add metadata to case data if it exists
        output_data = {
            "metadata": metadata,
            "case_data": case_data
        } if metadata else case_data

        # Write case to file
        try:
            with open(case_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Created case file: {case_file}")
        except Exception as e:
            print(f"❌ Error writing case file {case_id}: {e}")

def main():
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python split.py <path_to_json_file>")
        print("Example: python split.py graded/all_cases_results_5cases_20240306_123456.json")
        return

    json_path = sys.argv[1]

    # Verify file exists
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return

    # Verify it's a JSON file
    if not json_path.endswith('.json'):
        print(f"❌ File must be a JSON file: {json_path}")
        return

    # Get output folder name from user
    while True:
        output_name = input("Enter name for output folder: ").strip()
        if output_name:
            break
        print("❌ Folder name cannot be empty")

    print(f"🔄 Processing file: {json_path}")
    split_json_file(json_path, output_name)
    print("✨ Done!")

if __name__ == "__main__":
    main() 