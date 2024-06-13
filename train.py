import os
from covvec import create_coverage_vectors 

def each_file():
    # Specify the directory path
    directory_path = os.path.join(os.getcwd(), "tracks")
    
    all_data = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            all_data.append( create_coverage_vectors( feature_file=filename ) )
        else:
            print(f"Found directory: {filename}")

    all_data = np.array( all_data )
    return all_data

each_file()
