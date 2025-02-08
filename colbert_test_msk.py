'''
Takes a data file with columns corresponding to text and labels (see example ____)

Arguments:
--i Filename (data file)
--tc Name of text column
--p Text for prefix (prompting)
--om Name of output metrics file
--os Name of output scores file

Returns metrics using the specified ColBERT model

For example:
python colbert_test_msk.py --i medications/specific_prior_meds_with_notes_11_26_2024.csv --tc CDO_VALUE_TEXT --p "prior " --om ColBERT_priortx_metrics.csv --os specific_prior_meds_ColBERT_scores_11_26_2024.csv


# ***TROUBLESHOOTING TIPS***
# to work on CDSI cluster I had to do the following:
# 1. Activate colbert conda environment and run all of the following from there:
# 2. conda install ninja
# 3. Run the following:
CONDA_PREFIX=$(conda info --base)/envs/colbert

export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python setup.py clean
python setup.py build
# 4. Modify your colbert/indexing/codecs/residual.py to use load with the correct paths:
from torch.utils.cpp_extension import load

class ResidualCodec:
    @staticmethod
    def try_load_torch_extensions(use_gpu):
        CONDA_PREFIX = os.getenv('CONDA_PREFIX', '/path/to/your/conda/env')
        
# I also commented out the decompress_residuals_cpp = load( ... ) function call
'''

import colbert

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

import multiprocessing as mp

from sklearn import metrics
from sklearn.model_selection import train_test_split

import pandas as pd
#from mind_minio_client import client
import matplotlib.pyplot as plt
import numpy as np

import sys
from pre_post_processing import *


def main():

    # extract kwargs
    arguments = sys.argv[1:]
    kwargs = {}
    for i in range(0, len(arguments), 2):
        if len(arguments) > i + 1:
            kwargs[arguments[i].lstrip("--")] = arguments[i + 1]

    # INPUT DATA
#    obj = client.get_object('cdm-data',kwargs['i'])
#    text_reports = pd.read_csv(obj)
    text_reports = pd.read_csv(kwargs['i'])
    text_reports[kwargs['tc']]=text_reports[kwargs['tc']].fillna('').astype(str)+' ' # ensure no empty notes

  #  split into test/train
  #  np.random.seed(42)
  #  text_reports['val'] = np.random.choice([0, 1], size=len(text_reports), p=[0.8, 0.2])
  #  text_reports = text_reports[text_reports['val'].astype(bool)].reset_index()

    # PRE-PROCESSING
    collection = list(text_reports[kwargs['tc']].values)
    dataset = 'bpc'
    datasplit = 'dev'

    qthresh=5
    common_queries = []
    for c in list(text_reports.columns):
        if pd.api.types.is_bool_dtype(text_reports[c]):
            if sum(text_reports[c])>qthresh:
                common_queries+= [c]

    # expand queries for chemotherapy generic/brand and regimen names
    common_queries_full = augmentQueries(common_queries)

    # ColBERT indexing of documents
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300 # truncate passages at 300 tokens
    max_id = 10000

    index_name = f'{dataset}.{datasplit}.{nbits}bits'

    checkpoint = 'colbert-ir/colbertv2.0' #".ragatouille/colbert/none/2025-01/19/21.36.28/checkpoints/colbert/" #'.ragatouille/colbert/none/2024-07/16/14.43.02/checkpoints/colbert' #'colbert-ir/colbertv2.0'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # **4 kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
										# Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)

    # To create the searcher using its relative name (i.e., not a full path), set
    # experiment=value_used_for_indexing in the RunConfig.
    with Run().context(RunConfig(experiment='notebook')):
        config = ColBERTConfig(ncells=1024,centroid_score_threshold=0, ndocs=2**31 - 1, ignore_scores=True)
        searcher = Searcher(index=index_name, collection=collection, config=config)

    # RESULTS
    # loop through queries and store scores and metrics
    metrics_all_queries = pd.DataFrame(columns=['query', 'auc', 'optimal_threshold', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives'])
    for i, q in zip(common_queries,common_queries_full):
        query = kwargs['p']+q #'prior '+q #'tumor in '+q
    
        gold_standard = text_reports[i]
    
        print(f"#> {query}")
        results = searcher.search(query, k=len(text_reports)) # <-- replace this with an LLM classifier

        text_reports.loc[results[0],q+' ColBERT_score']=results[2]

        metrics = find_optimal_threshold_and_metrics(gold_standard, text_reports[q+' ColBERT_score'].fillna(0))
        metrics['query'] = i
        metrics_all_queries = pd.concat([metrics_all_queries, pd.DataFrame([metrics])], ignore_index=True)

    metrics_all_queries.to_csv(kwargs['om'],index=False)
    text_reports.to_csv(kwargs['os'],index=False)
    print('Done! Stored metrics in '+kwargs['os'])

if __name__ == '__main__':
    # This check is required to make multiprocessing work correctly
    mp.freeze_support()
    main()
