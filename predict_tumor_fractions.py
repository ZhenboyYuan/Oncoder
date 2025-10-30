import Oncoder
import torch
import pandas as pd
from Oncoder import Autoencoder
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

def predict(file,model):
    testdata = pd.read_csv(file,sep='\t',index_col=0)
    sample_names = testdata.columns

    print(f"Loading model from: {model}")
    model = torch.load(model,map_location=device,weights_only = False)
    testdata = testdata.T.values
    pred,_ = Oncoder.predict(testdata,model)
    
    res_df = pd.DataFrame(
        data=pred,
        index=sample_names,
        columns=['normal_fraction', 'tumor_fraction'])
    return res_df

def main():
    parser = argparse.ArgumentParser(description="Accepts the raw input test sets and predicts the tumor fractions from samples with different states")
    parser.add_argument('--file', type=str, required=True, help='Path to input test data')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--outfile', type=str, required=True, help='Path to out data')

    args = parser.parse_args()
    input_filepath = args.file
    input_modelpath = args.model
    outfile = args.outfile
    res = predict(input_filepath, input_modelpath)
    
    print(f"Saving results to: {outfile}")
    res.to_csv(outfile)
    print("File saved.")


if __name__ == "__main__":
    main()
