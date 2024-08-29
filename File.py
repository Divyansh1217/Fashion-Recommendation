from zipfile import ZipFile
from functools import cache
import os
@cache
def File_im():
    zip_file_path = r'C:\Users\divya\OneDrive\Desktop\okk\Fasion Recomm\women-fashion.zip'
    extraction_directory = r'C:\Users\divya\OneDrive\Desktop\okk\Fasion Recomm\women-fashion'

    if not os.path.exists(extraction_directory):
        os.makedirs(extraction_directory)

    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_directory)

    extracted_files = os.listdir(extraction_directory)
    print(extracted_files[:10])


    extraction_directory_updated = os.path.join(extraction_directory, 'women fashion')

    extracted_files_updated = os.listdir(extraction_directory_updated)
    return extraction_directory_updated,extracted_files_updated


if __name__=="__main__":
    path,files=File_im()
