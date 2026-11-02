# gpt-4.1 runs in the private environment

Two configs, same items (decisions.md #35):

| Config | Items per dataset | Calls for gpt-4.1, three datasets |
|---|---|---|
| `configs/pilot_1000.yaml` (with `-m gpt-4.1`) | 200 dev, 200 calibration, 600 eval | 123,500 |
| `configs/full_gpt41.yaml` (paper run) | every distinct question, 2:2:6 | about 1,050,000 |

The pilot items are the first items of each paper-run set, so the paper run reuses every cached
pilot call. The paraphrases and wrong candidate answers for the pilot are already made and come
with the code (`data/generated/generator.sqlite`), so every model sees the same items and the
pilot makes no generator calls. The paper run needs `pregenerate.py` run first on one machine
and the file committed. The earlier pilot ran about 7 calls per second on 8 workers on the direct API: about
5 hours for the pilot and about 42 hours for the paper run. SafeChain's speed is untested at
this scale.

Below, `CONFIG` is one of the two, for example:

```bash
CONFIG="configs/pilot_1000.yaml -m gpt-4.1"
```

## In the private environment

1. Get the code and install it:

   ```bash
   git clone -b main https://github.com/Michellemingxuan/confledger.git
   cd confledger
   pip install -e ".[data]"
   ```

   SafeChain and LangChain must already be installed in that environment.

2. Turn on the transport, and keep this environment's cache in its own folder:

   ```bash
   export LLM_BACKEND=safechain
   export CONFLEDGER_CACHE=experiments/results/cache_private
   ```

3. Run the preflight. It checks the three datasets and sends one short call to each model.

   ```bash
   python experiments/scripts/preflight.py -c $CONFIG
   ```

   - Dataset fails: the environment cannot reach Hugging Face. Copy in the dataset cache
     (`~/.cache/huggingface/datasets`) or point `HF_HOME` to one.
   - Judge fails: gpt-5.4 may not be served. Change the `judge:` line in the config to `gpt-5`
     and rerun. The judge can be replaced later with `recluster.py`.
   - gpt-4.1 or the generator fails: stop. The run cannot start.

4. Start collection in a session that survives a disconnect, such as `tmux`:

   ```bash
   python experiments/scripts/collect.py -c $CONFIG -v 2>&1 | tee collect.log
   ```

   If it stops, run the same command again. Finished items are skipped and cached calls are free.
   Each finished file logs `failures={...}`. In the earlier gpt-4.1 pilot, 16 answers failed to
   parse across 300 items, so a much higher rate means something is wrong.

5. Check the item counts. The pilot has 200 dev, 600 eval and 200 calibration items per dataset:

   ```bash
   wc -l experiments/results/*/*/collections/*.jsonl
   ```

## Bring the data back

First confirm that the collections may leave the private environment. They hold the questions,
gpt-4.1's answers and explanations, and the gold answers. For the pilot:

```bash
tar -czf gpt41_collections.tar.gz \
  experiments/results/*/pilot_1000/collections \
  experiments/results/all/pilot_1000/run_log.jsonl \
  experiments/results/cache_private
```

For the paper run, use `full_gpt41` in place of `pilot_1000`. `cache_private` is optional for
scoring, because evaluation reads item ids only. Bring it to keep the raw responses (for
`dev_audit.py`) and the paraphrases and distractors.

If nothing can leave, run the analysis there and export the reports instead:

```bash
experiments/scripts/run_all.sh configs/pilot_1000.yaml
```

## Analyse on the Mac

From the repo root, with `LLM_BACKEND` unset, unpack over the repo. Nothing else is overwritten:
the collections sit under the config's own name and the cache in its own folder.

```bash
tar -xzf gpt41_collections.tar.gz
python experiments/scripts/evaluate.py -c configs/pilot_1000.yaml -m gpt-4.1
```

Then run any paper script with the same flags. None of this calls an API.
