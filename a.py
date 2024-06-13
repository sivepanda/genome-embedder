import pybedtools

def create_coverage_vectors( feature, genome="hg38", window_size=1000 ):
    base = "tracks"
    genome_windows = pybedtools.BedTool().window_maker(g=os.path.join(os.getcwd(), base , genome + '.fa.fai') , w=window_size)
    feature = pybedtools.example_bedtool(os.path.join(os.getcwd(), base , "knownAlt.bed"))

    overlap = genome_windows.intersect(feature, wao=True)
    

    coverage_vec = []

    for f in overlap:
        try:
            n = int(f[-1])
        except:
            n = 0
        coverage_vec.append(n / window_size)

    return coverage_vec

    # sum(f.length for f in (feature.intersect(reference, u=True))) / (sum(f.length for f in feature))
