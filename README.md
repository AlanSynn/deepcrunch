# DeepCrunch
---

LG U+ MLOps Internship
Model Compressor for General Usage

Created by: Alan Synn (alansynn@gatech.edu)

---

## Set up

### Install dependencies

Create a conda environment to run deepcrunch as follows:

```
conda env create -f environment.yml -p ./env
```

## Code formatting

### Get linting results
```
conda activate ./env
make lint
```

### Automatically format code
```
conda activate ./env
make format
```

## Build

### Build Python package for development
```
conda activate ./env
make build-dev
```

### Build Python package
```
conda activate ./env
make build
```

### Install Python package
```
make install
```

### Appendix: Clean Build
```
conda activate ./env
make clean-build
```

### Appendix: reinstall with clean build
```
conda activate ./env
make reinstall
```

---

## Logistics

### VCS

+ Github
    + Create as private and send invitations
    + Necessary information for operations
        + Connect to the PC next to you
            + Connection information
            + SSH connection

### Documentation

1. Reports should be in Confluence format
2. Build manuals in a `docs` fashion

### Testing

1. Focus mainly on open models
    + e.g. Vicuna - 5B, ResNet-152
    + Discuss when the package is out
    + Revisit in the 4th week of July
+ The format should be similar to previous Confluence materials
    + [Confluence Reference](https://confluence.dx.lguplus.co.kr/pages/viewpage.action?pageId=179265360)

### Outcome

+ Midterm presentation in the third week
    + Presentation of progress
        + Design (UX) --> 50%
        + Overall library structure --> 50%
        + Plan
        + Technical aspects (anticipated performance or midterm presentation) -- guesstimate

+ Brief briefing on trends or research
    + More focused on trending things --> check later
        + Retraining

+ In the form of a library for `pip install`
    + Expandability
        + Compression algorithms (in-training/inference)
        + Scheduler (in training)
        + Accelerator (CPU/GPU)
        + If there is more time: multi-gpu serving
        + [PyTorch Pipeline Reference](https://pytorch.org/docs/stable/pipeline.html#pipeline-parallelism)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
