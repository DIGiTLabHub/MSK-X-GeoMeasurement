# Release validation

The preparation checks for this release completed successfully:

- Parsed all retained Python source modules.
- Five statistical unit checks and four input-integrity checks passed, including unequal patient-group weighting and vectorized-versus-literal bootstrap agreement.
- Recomputed 10,000-resample results from saved predictions: all 432 checked reference fields matched (absolute tolerance 0.0000002).
- Passed 525 additional checks of case references, stored per-seed metrics, recovered optimization records, exact prompts, inventory counts and retained scientific function definitions against the pre-cleanup release.
- Verified every released execution-script hash in `source_manifest.json` and every current protocol-table script path.
- Checked immutable reference hashes, split sizes and zero HVAngleEst patient overlap.
- Audited the downloaded datasets from a clean workspace containing only the public data and no old experiment outputs.
- Constructed the three declared direct architectures in dry-run mode on both datasets, without downloading pretrained weights or training.
- Validated the full 73 Cobb / 176 HVA / 176 IMA target lists for each prompt provider in dry-run mode; no provider request was sent.
- Regenerated the matched-prompt distribution and the seed-42 foot-axis/error figure.
- Verified local documentation links and scanned release contents for credentials, private keys, machine-specific paths and unexpected binary files; none were found.

Training losses, network definitions, geometric calculations and provider-call functions were compared with the executed-source baseline. Their function bodies were unchanged. The cleanup also removes non-paper branches and strengthens input validation. Retained training, feature, geometry, and provider computations are unchanged; see `source_manifest.json`.

These checks establish packaging and saved-record numerical reproducibility. Full GPU retraining and new provider interactions were not run during code preparation. Frozen-model extraction requires the upstream implementations and weights described in MODELS.md.
