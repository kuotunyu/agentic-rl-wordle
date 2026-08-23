# Data governance and split audit

## Source manifest

The repository does **not** redistribute either cfreshman word list. `scripts/fetch_words.py`
downloads revision-pinned text, normalizes it to unique lowercase ASCII five-letter words, and
rejects content whose count or normalized SHA-256 differs from this manifest.

| Role | Source revision | Expected count | Normalized SHA-256 | License status |
|---|---|---:|---|---|
| answers | cfreshman gist `a03ef…` @ `c46f451920d5cf6326d550fb2d6abb1642717852` | 2,315 | `5209b35f823f8b80f0404f863bd80df06d6a966c6eb1016d69f38badc6eed5d0` | No explicit license found in the gist |
| extra allowed guesses | cfreshman gist `cdcdf…` @ `d7c9e02d45afd26e12a71b4564189a949c29e8a9` | 10,657 | `99be2e38dadf3e26952af7cb4d963f65b632d5de91aa99e5ce308e4dc9617b65` | No explicit license found in the gist |
| legal-list fallback | tabatkins/wordle-list @ `255b9469c4dad99a3b95cc4ddbe139b3d3747868` | 14,855 | `a7898cd20f36686d4c5b43ece226c36eb41a701df36da35bd18e21af41cfead4` | MIT |

“Publicly reachable” is not treated as “openly licensed.” The Apache-2.0 repository license
applies to this repository's code and documentation, not to third-party word-list content.
External publication must keep the lists fetch-only, retain attribution, and preserve this
license caveat. A publisher who needs stronger dataset rights should replace the answer list
with a clearly licensed equivalent and rerun training/evaluation rather than silently changing
the existing evidence.

## Deterministic split

`wordle_rl.words.get_splits()` sorts the 2,315 answers, shuffles them with Python
`random.Random(42)`, and takes the first `int(2315 * 0.8) = 1,852` for training. The remaining
463 form `eval_full`; its first 200 are the earlier nested interim look `eval_200`.

The committed executable checks establish:

- answers: 2,315; extra allowed: 10,657; legal union: 12,972;
- train/eval sizes: 1,852/463;
- train/eval intersection: empty;
- train/eval union: all 2,315 answers;
- seed 42 is deterministic and seed 7 produces a different split;
- `eval_200` is a prefix of `eval_full`, not an independent replication.

Evidence: `scripts/fetch_words.py`, `src/wordle_rl/words.py`, `tests/test_fetch_words.py`, and
`tests/test_words.py`.

## Repository boundary

Generated `data/answers.txt`, `data/allowed.txt`, and `data/SOURCE.json` are ignored. Only
`data/.gitkeep` is published. `SOURCE.json` records the exact URLs, revisions, normalized hashes,
counts, and license-status label used by a local fetch.
