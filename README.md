# onc_IR_LLM
code for combining information retrieval (IR, i.e. ColBERT) and LLMs for annotating oncologic EHR datasets

## Dependencies


## Instructions for use (IR portion)
1. Install [ColBERT](https://github.com/stanford-futuredata/ColBERT) according to the conda installation guide on the git repo
2. For metrics will require sklearn as well
3. Run colbert_test_msk.py (see Troubleshooting tips below)
4. Visualize results using analyze_metrics.ipynb

## ***TROUBLESHOOTING TIPS***
to work on CDSI cluster I had to do the following:
1. Activate colbert conda environment and run all of the following from there:
2. conda install ninja
3. Run the following:
CONDA_PREFIX=$(conda info --base)/envs/colbert

export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python setup.py clean
python setup.py build
4. Modify your colbert/indexing/codecs/residual.py to use load with the correct paths:
from torch.utils.cpp_extension import load

class ResidualCodec:
    @staticmethod
    def try_load_torch_extensions(use_gpu):
        CONDA_PREFIX = os.getenv('CONDA_PREFIX', '/path/to/your/conda/env')
        
I also commented out the decompress_residuals_cpp = load( ... ) function call
