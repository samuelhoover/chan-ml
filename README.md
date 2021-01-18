# chan-ml

chan-ml was written with Python 3.7 but should be compatible with other Python 3 versions. The scripts in the base directory i.e. `chan-ml` are not project specific and can be used for any neural network operation with minimal tweaking.

To make sure all necessary packages and package versions are installed, do the following depending on which package manager you use: 
  * If using pip:
    
    - From Python 3.7 environment, run `pip -r install requirements_pip.txt`.
    
  * If using Conda:
    
    - Run `conda create --name <your_env_name> --file requirements_conda.txt` to create a Python 3.7 environment with all the necessary packages.

Run `./run_chan-ml.sh` to start training or predicting. Make sure your `gen_args.txt`, `train_args.txt`, and `predict_args.txt` argument files are configured correctly before beginning.

May need to run `chmod +x run_chan-ml.sh` to change file permissions and make Bash script executable.
