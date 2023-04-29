# Setup
```
conda env create -f environment.yml
```

This creates the snowflakes environment

```
conda activate snowflakes
```

This activates the environment

# Adding new packages / dependencies
Use `conda install` and then follow the section `Saving environment` to export to the environment.yml file

# Saving environment
```
conda env export > environment.yml
```
# Match conda environment to Python
search "Python Interpreter" and click the version with "snowflakes"