STORAGE_ROOT = /media/steve/storage

# ========== Dataset Preparation ==========

# ARC-AGI-1
get-arc1:
	python -m dataset.build_arc_dataset \
		--input-file-prefix kaggle/combined/arc-agi \
		--output-dir $(STORAGE_ROOT)/data/arc1concept-aug-1000 \
		--subsets training evaluation concept \
		--test-set-name evaluation

# ARC-AGI-2
get-arc2:
	python -m dataset.build_arc_dataset \
		--input-file-prefix kaggle/combined/arc-agi \
		--output-dir $(STORAGE_ROOT)/data/arc2concept-aug-1000 \
		--subsets training2 evaluation2 concept \
		--test-set-name evaluation2

# Sudoku-Extreme
get-sudoku:
	python dataset/build_sudoku_dataset.py \
		--output-dir $(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000 \
		--subsample-size 1000 \
		--num-aug 1000

# Maze-Hard
get-maze:
	python dataset/build_maze_dataset.py \
		--output-dir $(STORAGE_ROOT)/data/maze-30x30-hard-1k

# Prepare all datasets
prepare-data: get-arc1 get-arc2 get-sudoku get-maze

# ========== Training ==========

# Train ARC-AGI-1 (single GPU)
train-arc1:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/arc1concept-aug-1000]" \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=4 \
		+run_name=pretrain_att_arc1concept_1gpu \
		ema=True

# Train ARC-AGI-2 (single GPU)
train-arc2:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/arc2concept-aug-1000]" \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=4 \
		+run_name=pretrain_att_arc2concept_1gpu \
		ema=True

# Train Sudoku-Extreme with MLP (single GPU)
train-sudoku-mlp:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000]" \
		evaluators="[]" \
		epochs=50000 eval_interval=5000 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.mlp_t=True arch.pos_encodings=none \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=6 \
		+run_name=pretrain_mlp_t_sudoku_1gpu \
		ema=True

# Train Sudoku-Extreme with Attention (single GPU)
train-sudoku:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000]" \
		evaluators="[]" \
		epochs=50000 eval_interval=5000 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=6 \
		+run_name=pretrain_att_sudoku_1gpu \
		ema=True

# Train Maze-Hard (single GPU)
train-maze:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/maze-30x30-hard-1k]" \
		evaluators="[]" \
		epochs=50000 eval_interval=5000 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=4 \
		+run_name=pretrain_att_maze30x30_1gpu \
		ema=True
