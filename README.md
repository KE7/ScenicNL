# scenicNL

This repo serves as a wrapper around [Scenic](https://github.com/BerkeleyLearnVerify/Scenic) where user's can specifiy a file containing natural language text describing a driving scene. Our contribution is that we take this natural language text and output scenic code that represent the scenario.

At the moment we only support driving scenarios for CARLA.

## Setup
To install and run, first set up OpenAI API key if you have not already:

1. Login to OpenAI account and go to `https://platform.openai.com/account/api-keys`.
2. Create an API key.
3. Now open a shell: on Windows run `set OPENAI_API_KEY=<your-key>`, and on Unix run `export OPENAI_API_KEY=<your-key>`.
   We will also need an organization id so also `OPENAI_ORGANIZATION=<your-org-id>` or `export OPENAI_ORGANIZATION=<your-org-id>`

```bash
conda create -n scenicNL python=3.11
conda activate scenicNL
pip install -e '.[dev]'
```

An example command to running our pipeline
```bash
gen_scenic --query_path <path-to-descriptions> --output_path <path-to-output> --model gpt-3.5-turbo-0613 --prompt_type predict_few_shot
```