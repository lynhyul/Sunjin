import os
import shutil

compare_folder = 'D:/yolo/data/06_14/labels/'
base_folder= 'D:/yolo/data/06_14/images/'


def remove_extra_files(base_folder, compare_folder):
    # Get the list of text files in the base folder
    base_files = [f for f in os.listdir(base_folder) if f.endswith('.jpg')]

    # Get the list of image files in the compare folder
    compare_files = [f for f in os.listdir(compare_folder) if f.endswith('.txt')]

    # Iterate over the text files in the base folder
    for file_name in base_files:
        file_path = os.path.join(compare_folder, file_name)

        # Check if the corresponding image file exists in the compare folder
        image_file = file_name.replace('.txt', '.jpg')  # Modify the extension as per your image file format
        if image_file not in compare_files:
            # Remove the text file from the base folder
            os.remove(os.path.join(base_folder, file_name))
            print(f"Deleted file: {os.path.join(base_folder, file_name)}")


# Call the function to remove extra files
remove_extra_files(base_folder, compare_folder)