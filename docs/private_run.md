# Full gpt-4.1 run in the private environment

`configs/full_gpt41.yaml`: TriviaQA, StrategyQA and MMLU, 800 items each (100 dev, 500 eval,
200 calibration). 98,800 calls: 93,600 to gpt-4.1, 2,400 to the judge (gpt-5.4), 2,800 to the
generator (gpt-5). The pilot ran about 7 calls per second on 8 workers on the direct API, which
would be about 4 hours here. SafeChain's speed is untested at this scale.

## In the private environment

1. Get the code and install it:

   ```bash
   git clone -b cel-k-safeguards-private-run https://github.com/Michellemingxuan/confledger.git
   cd confledger
   pip install -e ".[data]"
   ```

   SafeChain and LangChain must already be installed in that environment.

2. Turn on the transport:

   ```bash
   export LLM_BACKEND=safechain
   ```

3. Run the preflight. It checks the three datasets and sends one short call to each model.

   ```bash
   python experiments/scripts/preflight.py -c configs/full_gpt41.yaml
   ```

   - Dataset fails: the environment cannot reach Hugging Face. Copy in the dataset cache
     (`~/.cache/huggingface/datasets`) or point `HF_HOME` to one.
   - Judge fails: gpt-5.4 may not be served. Change the `judge:` line in the config to `gpt-5`
     and rerun. The judge can be replaced later with `recluster.py`.
   - gpt-4.1 or the generator fails: stop. The run cannot start.

4. Start collection in a session that survives a disconnect, such as `tmux`:

   ```bash
   python experiments/scripts/collect.py -c configs/full_gpt41.yaml -v 2>&1 | tee collect.log
   ```

   If it stops, run the same command again. Finished items are skipped and cached calls are free.
   Each finished file logs `failures={...}`. In the gpt-4.1 pilot, 16 answers failed to parse
   across 300 items, so a much higher rate means something is wrong.

5. Check that all nine collections have the expected item counts (dev 100, eval 500, cal 200):

   ```bash
   wc -l experiments/results/*/full_gpt41/collections/*.jsonl
   ```

## Bring the data back

First confirm that the collections may leave the private environment. They hold the questions,
gpt-4.1's answers and explanations, and the gold answers.

```bash
tar -czf full_gpt41_collections.tar.gz \
  experiments/results/*/full_gpt41/collections \
  experiments/results/cache_full \
  experiments/results/all/full_gpt41/run_log.jsonl
```

`cache_full` is required: `evaluate.py` rebuilds the TriviaQA items from the cached distractors.

If nothing can leave, run the analysis there and export the reports instead:

```bash
experiments/scripts/run_all.sh configs/full_gpt41.yaml
```

## Analyse on the Mac

From the repo root, unset `LLM_BACKEND` and unpack over the repo. Nothing else is overwritten,
because the run has its own config name and cache folder.

```bash
tar -xzf full_gpt41_collections.tar.gz
python experiments/scripts/evaluate.py -c configs/full_gpt41.yaml
```

Then run any paper script with the same `-c` flag. None of this calls gpt-4.1. A generator call
during `evaluate.py` means the cache did not arrive intact, so stop and check `cache_full`.
