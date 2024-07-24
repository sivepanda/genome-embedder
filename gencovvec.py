import os 
import pybedtools
import numpy as np
import concurrent.futures
import threading
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=np.inf)

# Return a numpy array of arrays for every file in coverage_vector_path
def get_coverage_vectors(track_path='tracks', coverage_vector_path='covvecs', new_coverage_vectors=False):
    all_data = []

    if new_coverage_vectors == True:
        coverage_vectors_from_bed( track_path=track_path, coverage_vector_path=coverage_vector_path )
    dirlen = (len(os.listdir(coverage_vector_path)))
    i = 0
    for filename in os.listdir( coverage_vector_path ):
        i += 1
        print(f'Opening file {i} / {dirlen}: {filename}                                             ', flush=True, end='\r')
        file_path = os.path.join(coverage_vector_path , filename)
        if os.path.isfile(file_path):
            temp = np.load(file_path)
            all_data.append(temp)
    print("\t")
    print("Constructing numpy array...")
    return  np.array( all_data ) 

# Return a LABELLED numpy array of arrays for every file in coverage_vector_path
def get_labelled_coverage_vectors( coverage_vector_path='covvecs' ):
    all_data = {}

    for filename in os.listdir( coverage_vector_path ):
        file_path = os.path.join(coverage_vector_path , filename)
        if os.path.isfile(file_path):
            temp = np.load(file_path)
            id_name = file_path.split("/")[1].split(".")[0]
            all_data[ id_name ] = temp

    return all_data


# Creates a numpy array for each BED file located in the track_path (./tracks by default). Saves the numpy arrays to files in the coverage_vector_path (./covvecs by default)
def coverage_vectors_from_bed(track_path = 'tracks', coverage_vector_path='covvecs'):
    track_path=os.path.join(os.getcwd(), track_path)
    coverage_vector_path = os.path.join(os.getcwd(), coverage_vector_path)
    print(track_path, coverage_vector_path)
    
    # Initialize multithreaded operation and read all files located in track_path
    max_workers = os.cpu_count() if os.cpu_count() < 2 else 2
    print("This program uses multithreaded operation. This program is using", max_workers, "threads.")
    print("Reading files in ./", track_path)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        p = os.path.join(os.getcwd(), coverage_vector_path)
        futures = [executor.submit( create_coverage_vector, feature_file=filename, save=True, path=os.path.join(p, filename.split(".")[0] )) for filename in os.listdir(track_path)]
        [future.result() for future in concurrent.futures.as_completed(futures)]
    return True

# Returns an hashmap of bedtools objects for each of the inputted features. Pull data from UCSC if desired.
def create_bedtools(genomic_features, base, genome, chrom, start, end):
    genomic_features_bedtools = {}

    print("Creating BEDTools objects...")
    for feature in genomic_features: 
        genomic_features_bedtools[feature] = pybedtools.example_bedtool(os.path.join(os.getcwd(), base , genome + "_" + feature + ".bed"))
    print("BEDTools created.")

    return genomic_features_bedtools

# Creates a vector containing the proportion of each window that is covered by a particular track
def create_coverage_vector( feature='', feature_file='', genome="hg38", base='tracks', window_size=1000, save=False, path=''):
    print(f"Thread {threading.get_ident()} is processing", feature_file)
    if len( feature ) == 0 and len( feature_file ) == 0:
        raise Exception("You must provide an argument for either feature or feature_file")
    if save==True and len( path ) == 0:
        raise Exception("Please specify a path to save the coverage vector to, or set save to false")
    if feature_file.split(".")[-1] != "bed":
        print("Skipped", feature_file ,"because it is not a BED file")
        return False
    if os.path.isfile(path + '.npy'):
        print(feature_file, "has already been processed")
        return True
    
    if len( feature_file ) == 0:
        feature_file = feature + '.bed'
    windows = pybedtools.BedTool().window_maker(g=os.path.join(os.getcwd(), base, 'ref', genome + '.fa.fai') , w=window_size)
    feature = pybedtools.example_bedtool(os.path.join(os.getcwd(), base , feature_file))
    overlap = windows.coverage(feature)
    coverage_vec = []
    print(len(overlap))
    # for now, we'll omit the final sequences of each chromosome to allow for vectors to retain the same dimensionality
    # later, strategy could either be appending zeros to fill out the remainder of the window, or it could be adjusting 
    # window size to work with everything else, assuming it makes sense computationally + no prime lengths
    for f in overlap:
        if float( f[-1] ) < 0:
            print( f[-1] )
        try:
            n = float( f[-1] )
            if n <= 1 and n >= 0:
                coverage_vec.append(n)
        except:
            n = 0
            coverage_vec.append(n)
    
    ret = np.array( coverage_vec )

    if save:
        np.save( path, ret )
        print("Saving array to ", path)

    return len(ret) == window_size
