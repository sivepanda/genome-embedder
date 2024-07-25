import numpy as np
from gencovvec import get_labelled_coverage_vectors

# Encode values using a TensorFlow model
def get_encoded_vals_tf(model_file='./models/encoder.keras', verbose_output=False):
    # Load function-specific libaries
    from tensorflow.keras.models import load_model
    
    # Load TensorFlow model
    encoder = load_model(model_file)

    np.set_printoptions(threshold=np.inf)
    lbc = get_labelled_coverage_vectors()
    encoded_vals = {}
    for key, value in lbc.items():
        print(key)
        encoded_vals[ key ] = encoder.predict( np.array( [value] ) )
        if verbose_output:
            print( encoded_vals[ key ] )
    return encoded_vals



# Encode values using a PyTorch model
def get_encoded_vals_torch(model_file='./models/model.pth', verbose_output=False):
    # Load function-specific libaries
    import torch


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_location = torch.device("cpu")

    # Set print limit to infinity
    np.set_printoptions(threshold=np.inf)
    
    # Load PyTorch file
    model = torch.load(model_file) if device == "cuda" else torch.load(model_file, map_location=map_location) 
    model.eval()

    lbc = get_labelled_coverage_vectors()
    encoded_vals = {}
    for key, value in lbc.items():
        print(key)
        data = np.array( [ value ] )
        data = data.astype(np.float32)
        data = torch.tensor(data, device=device).unsqueeze(1)# .to(dtype=torch.float16)
        encoded_vals[ key ] = [ model( data.float() ).detach().cpu().numpy().squeeze() ]
        if verbose_output:
            print( encoded_vals[ key ] )
    return encoded_vals
