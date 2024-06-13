import os 
import pybedtools
import numpy as np


# Returns an hashmap of bedtools objects for each of the inputted features. Pull data from UCSC if desired.
def create_bedtools(genomic_features, base, genome, chrom, start, end):
    genomic_features_bedtools = {}

    print("Creating BEDTools objects...")
    for feature in genomic_features: 
        genomic_features_bedtools[feature] = pybedtools.example_bedtool(os.path.join(os.getcwd(), base , genome + "_" + feature + ".bed"))
    print("BEDTools created.")

    return genomic_features_bedtools

# Creates a vector containing the proportion of each window that is covered by a particular track
def create_coverage_vectors( feature='', feature_file='', genome="hg38", base='tracks', window_size=1000 ):
    if len( feature ) == 0 and len( feature_file ) == 0:
        raise Exception("You must provide an argument for either feature or feature_file")
    
    if len( feature_file ) == 0:
        feature_file = feature + '.bed'

    genome_windows = pybedtools.BedTool().window_maker(g=os.path.join(os.getcwd(), base , genome + '.fa.fai') , w=window_size)
    feature = pybedtools.example_bedtool(os.path.join(os.getcwd(), base , feature_file))

    overlap = genome_windows.intersect(feature, wao=True)

    coverage_vec = []

    for f in overlap:
        try:
            n = int(f[-1])
        except:
            n = 0
        coverage_vec.append(n / window_size)

    return np.array( coverage_vec )

    # sum(f.length for f in (feature.intersect(reference, u=True))) / (sum(f.length for f in feature))
    # return np.array(overlap_percentage.to_dataframe()['name'])


# overlap_percentage = overlap.apply(calculate_percentage)

