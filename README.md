<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_light.svg">
  <img alt="Zeus logo" width="55%" src="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_light.svg">
</picture>
<h1>Deep Learning Energy Measurement and Optimization</h1>

[![Slack workspace](https://badgen.net/badge/icon/Join%20workspace/b31b1b?icon=slack&label=Slack)](https://join.slack.com/t/zeus-ml/shared_invite/zt-2j5o12jqp-3LtNjgF_uBDTdNcaxWgpdw)
[![Docker Hub](https://badgen.net/docker/pulls/symbioticlab/zeus?icon=docker&label=Docker%20pulls)](https://hub.docker.com/r/symbioticlab/zeus)
[![Homepage](https://custom-icon-badges.demolab.com/badge/Homepage-ml.energy-23d175.svg?logo=home&logoColor=white&logoSource=feather)](https://ml.energy/zeus)
[![Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/ml-energy/zeus?logo=law)](/LICENSE)
</div>

---
**Project News** ⚡ 

- \[2024/08\] Perseus, an optimizer for large model training, was accepted to SOSP'24! [Paper](https://dl.acm.org/doi/10.1145/3694715.3695970) | [Blog](https://ml.energy/zeus/research_overview/perseus) | [Optimizer](https://ml.energy/zeus/optimize/pipeline_frequency_optimizer)
- \[2024/07\] Added AMD GPU, CPU, and DRAM energy measurement support, and preliminary JAX support!
- \[2024/05\] Zeus is now a PyTorch ecosystem project. Read the PyTorch blog post [here](https://pytorch.org/blog/zeus/)!
- \[2024/02\] Zeus was selected as a [2024 Mozilla Technology Fund awardee](https://foundation.mozilla.org/en/blog/open-source-AI-for-environmental-justice/)!
- \[2023/07\] We used the [`ZeusMonitor`](https://ml.energy/zeus/reference/monitor/energy/#zeus.monitor.energy.ZeusMonitor) to profile GPU time and energy consumption for the [ML.ENERGY leaderboard & Colosseum](https://ml.energy/leaderboard).
---

Zeus is a library for (1) [**measuring**](https://ml.energy/zeus/measure) the energy consumption of Deep Learning workloads and (2) [**optimizing**](https://ml.energy/zeus/optimize) their energy consumption.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).

## Repository Organization

```
zeus/
├── zeus/             # ⚡ Zeus Python package
│  ├── monitor/       #    - Energy and power measurement (programmatic & CLI)
│  ├── optimizer/     #    - Collection of time and energy optimizers
│  ├── device/        #    - Abstraction layer over CPU and GPU devices
│  ├── utils/         #    - Utility functions and classes
│  ├── _legacy/       #    - Legacy code to keep our research papers reproducible
│  ├── metric.py      #    - Prometheus metric export support
│  ├── show_env.py    #    - Installation & device detection verification script
│  └── callback.py    #    - Base class for callbacks during training
│
├── zeusd             # 🌩️ Zeus daemon
│
├── docker/           # 🐳 Dockerfiles and Docker Compose files
│
└── examples/         # 🛠️ Zeus usage examples
```

## Getting Started

Please refer to our [Getting Started](https://ml.energy/zeus/getting_started) page.
After that, you might look at

- [Measuring Energy](https://ml.energy/zeus/measure)
- [Optimizing Energy](https://ml.energy/zeus/optimize)

### Docker image

We provide a Docker image fully equipped with all dependencies and environments.
Refer to our [Docker Hub repository](https://hub.docker.com/r/mlenergy/zeus) and [`Dockerfile`](docker/Dockerfile).

### Examples

We provide working examples for integrating and running Zeus in the [`examples/`](/examples) directory.

## Research

Zeus is rooted on multiple research papers.
Even more research is ongoing, and Zeus will continue to expand and get better at what it's doing.

1. Zeus (NSDI 23): [Paper](https://www.usenix.org/conference/nsdi23/presentation/you) | [Blog](https://ml.energy/zeus/research_overview/zeus) | [Slides](https://www.usenix.org/system/files/nsdi23_slides_chung.pdf)
1. Chase (ICLR Workshop 23): [Paper](https://arxiv.org/abs/2303.02508)
1. Perseus (SOSP 24): [Paper](https://arxiv.org/abs/2312.06902) | [Blog](https://ml.energy/zeus/research_overview/perseus) | [Slides](https://jaewonchung.me/pdf.js/web/viewer.html?file=/assets/attachments/pubs/Perseus_slides.pdf#pagemode=none)

If you find Zeus relevant to your research, please consider citing:

```bibtex
@inproceedings{zeus-nsdi23,
    title     = {Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training},
    author    = {Jie You and Jae-Won Chung and Mosharaf Chowdhury},
    booktitle = {USENIX NSDI},
    year      = {2023}
}
```


## This fork is for integrating Energy Profiler in Zeus.

### Running Instructions Below

### 1. Create a **Dockerfile** in your project root:

```bash
FROM nvcr.io/nvidia/pytorch:24.02-py3
```

Build the Docker image with host networking:
```bash
docker build --network=host -t zeus-megatron .
```

### 2. Run the container, mounting your local repo:

```bash
docker run -it --rm \
  --ipc=host --shm-size=512m --gpus=4 \
  -v <path>/zeus:/workspace/zeus \
  -w /workspace/zeus \
  --cap-add SYS_ADMIN \
  --network=host \
  zeus-megatron:latest bash
```

### 3. Install dependencies and start the PFO server inside the container:
```bash
cd workspace/zeus
pip install -e ".[pfo, pfo-server]". # If this doesn't work, then the below steps one by one
pip install -e . --upgrade-strategy only-if-needed
pip install "pydantic<2"
pip install .[pfo, pfo-server] --no-deps
pip install aiofiles fastapi starlette uvicorn --no-deps
pip uninstall pynvml # because we use nvidia-ml-py
```

```bash
# Configure and launch the server on a separate docker root access
docker ps # for fetching name of your container - e.g. c8dab19234f
docker exec -it <your-zeus-container-name> bash  # this will lauch another terminal with access to same docker
export ZEUS_PFO_DUMP_DATA=true
export ZEUS_PFO_SCHEDULER=InstructionProfiler
export ZEUS_PFO_SCHEDULER_ARGS='{"solution_path": "/path/to/freqs_pipeline_%05d.py", "dump_dir": "/path/to/dump"}'

uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port 7787
```

<mark>The server will be accessible at http://127.0.0.1:7787.</mark>


### 4. Run model training pointing at the PFO server:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
  cd workspace/zeus/megatron
./run_training.sh --pfo-server-url "http://localhost:7787"
```

<mark>Once it runs, the csv file can be found in the dump directory, one sample file is already present there.</mark>

### 5. (Optional) Delete old checkpoints on the host:

HOST_DIR= < pathTo >/megatron/experiments
docker run --rm -v "$HOST_DIR":/mnt alpine sh -c 'rm -rf /mnt/codeparrot-small'



## Data Preparation

### 1. Download GPT‑2 vocabulary and merges:
```bash
cd /workspace/zeus/megatron
wget https://huggingface.co/gpt2/resolve/main/vocab.json
wget https://huggingface.co/gpt2/resolve/main/merges.txt
```

### 2. Subset the CodeParrot dataset:

#### This is alread available in megatron/get_data.py
```bash
from datasets import load_dataset

# Load only the first 2,000 samples of the training split
train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train[:2000]')

# Save the subset in JSON lines format
train_data.to_json("codeparrot_data.json", lines=True)
```

<mark>Run the data retrieval script (get_data.py):</mark>
```bash
python get_data.py
```

### 3. Preprocess CodeParrot content for training:
```bash
pip install nltk 

python tools/preprocess_data.py \
  --input codeparrot_data.json \
  --output-prefix codeparrot \
  --vocab-file vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file merges.txt \
  --json-keys content \
  --workers 32 \
  --append-eod
  
```

<mark>This outputs two files codeparrot_content_document.idx and codeparrot_content_document.bin which are used in the training. Reference: https://huggingface.co/blog/megatron-training#how-to-train-with-megatron-lm-</mark>



## Other Resources

1. Energy-Efficient Deep Learning with PyTorch and Zeus (PyTorch conference 2023): [Recording](https://youtu.be/veM3x9Lhw2A) | [Slides](https://ml.energy/assets/attachments/pytorch_conf_2023_slides.pdf)

## Contact

Jae-Won Chung (jwnchung@umich.edu)
