#!/bin/bash

chmod +x cache_latents.py

# Generate Latents
python cache_latents.py --csv train.csv --out_dir latents/train

python cache_latents.py --csv dev.csv   --out_dir latents/dev

python cache_latents.py --csv test.csv   --out_dir latents/test
