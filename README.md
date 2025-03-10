
## Running the experiments
### requirements

Dependencies (with python >= 3.9): Main dependencies are torch==2.0.0 torch-cluster==1.6.1 torch-geometric==2.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-spline-conv==1.2.2 

Commands to install all the dependencies in a new conda environment

```
conda create --name RIdiffusion python=3.10
conda activate RIdiffusion

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

### Dataset and Preprocessing

you can get rnasolo dataset on https://rnasolo.cs.put.poznan.pl/

```
python generate_graph_ss.py
```


### Experiments

```
python seq_generator.py
```
