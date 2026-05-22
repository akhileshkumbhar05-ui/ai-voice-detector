import os
import shutil

def flatten_and_rename_files(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 1  # Start counter

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_filename = f"elab_{count:04d}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(output_dir, new_filename)

            # Copy the file (use shutil.move() instead if you want to move)
            shutil.copy2(old_path, new_path)

            print(f"Copied: {old_path} -> {new_path}")
            count += 1

    print(f"\nDone. {count-1} files processed.")

# Example usage:
root_folder = "data/raw/ai_mp3"       # Replace with your source folder
output_folder = "data/raw/ai"   # Replace with where you want all files

flatten_and_rename_files(root_folder, output_folder)
