# ScenicNL

The Compound AI System that can generate [Scenic](https://github.com/BerkeleyLearnVerify/Scenic) programs from Crash Report descriptions.

At the moment we only support driving scenarios for CARLA but this is easily extendable as Scenic supports other simulators.

## Setup
To install and run, first select which LLM backbone you would like to use. 
Today we support GPT from OpenAI, Claude from Anthropic, and Open Source Models (i.e. Llama family) that are run locally.

### OpenAI GPT Steps
OpenAI API key if you have not already:
1. Login to OpenAI account and go to `https://platform.openai.com/account/api-keys`.
2. Create an API key.
3. Now open a shell: on Windows run `set OPENAI_API_KEY=<your-key>`, and on Unix run `export OPENAI_API_KEY=<your-key>`.
   We will also need an organization id so also `OPENAI_ORGANIZATION=<your-org-id>` or `export OPENAI_ORGANIZATION=<your-org-id>`

### Local Model Steps
One good option for a local model, is the Mixtral Mixture of Experts model from Mistral AI.
While you can run mixtral just fine through `llama.cpp`, we found in our experience, that it's just a bit faster
to use [Mozilla's Llamafile](https://github.com/Mozilla-Ocho/llamafile). Our code does not make any assumption 
on which local model you choose to run from there but we found the Mixtral-8x7B-Instruct works great for our use case.
Once you have picked which model to download from there list of supported models, follow the first 4 step's 
[quick start guide](https://github.com/Mozilla-Ocho/llamafile) to almost get running.
For step 5 when you are about to run the `llamafile`, also add the arguments: `--server -np 10` where `np` is the 
number of server threads (so 10 in this case) which should be **greater than or equal to** the number of worker threads you will use 
in ScenicNL.

Full example command:
`./mixtral-8x7b-instruct-v0.1.Q5_K_M.llamafile --server -np 10`


## Steps to install and run ScenicNL
```bash
conda create -n scenicNL python=3.11
conda activate scenicNL
pip install -e .
```

An example command to running our pipeline
```bash
gen_scenic --query_path <path-to-descriptions> --output-path <path-to-output> --model gpt-3.5-turbo-0613 --llm_prompt_type predict_few_shot
```
