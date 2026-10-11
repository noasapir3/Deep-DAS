Scripts of the targeted-data study, archived exactly as they were run (byte-identical copies).

  x_train_eval_w.py                      training + evaluation of one W1/W2 seed (the run that produced ../../models/W1, W2)
  x_train_eval_s__final_model_recipe.py  the training recipe of DeepDAS Ensemble v1 that x_train_eval_w.py extends
                                         (the line-level difference is in ../../records/*DIFF_x_train_eval_w_vs_s.txt)
  w00_hash_guard.py                      hash guard run before and after the round
  w01_lib.py, w05_import.py              helpers and the import of the round's simulations

They refer to the directory layout of the original working tree (paths relative to the project's DL folder) and are NOT
runnable from this release as they are. They are kept as the authoritative record of what was executed.

Re-running the W1/W2 training was not part of the release verification (only the evaluation and selection, via
../evaluate_selection_rule.py, which reproduces all seven result files byte for byte from the shipped checkpoints and data).
The release's own, verified training code is src/deepdas/train/supervised.py (DeepDAS Ensemble v1 recipe).
