import numpy as np
import pandas as pd
 
def npy_to_csv(npy_file, csv_file):
    
    data = np.load(npy_file, allow_pickle=True)
   
    
    df = pd.DataFrame(data)
   
    
    df.to_csv(csv_file, index=False)
    print(f"Successfully converted {npy_file} to {csv_file}")
 

npy_file = 'final_combined_signatures.npy'
csv_file = 'combined_signatures.csv'
npy_to_csv(npy_file, csv_file)