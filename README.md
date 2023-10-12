# scenicNL

This repo serves as a wrapper around [Scenic](https://github.com/BerkeleyLearnVerify/Scenic) where user's can specifiy a file containing natural language text describing a driving scene. Our contribution is that we take this natural language text and output scenic code that represent the scenario.

## Setup
```bash
conda create -n scenicNL python=3.11
conda activate scenicNL
pip install -e '.[dev]'
```

At the moment we only support driving scenarios for CARLA.