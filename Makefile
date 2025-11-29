STORAGE_ROOT = /media/steve/storage

.PHONY: get-arc1 get-arc2 get-sudoku get-maze prepare-data
.PHONY: train-arc1 train-arc2 train-sudoku train-sudoku-mlp train-maze
.PHONY: train-arc1-low-mem train-sudoku-low-mem

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
		--output-dir $(STORAGE_ROOT)/data/maze-30x30-hard-1k \
		--aug

# Prepare all datasets
prepare-data: get-arc1 get-arc2 get-sudoku get-maze

# ========== Training (Memory-Optimized for RTX 4070 12GB) ==========

# Train ARC-AGI-1 (single GPU, memory-optimized)
train-arc1:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/arc1concept-aug-1000]" \
		global_batch_size=32 \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=4 \
		arch.halt_max_steps=8 \
		+run_name=pretrain_att_arc1concept_1gpu \
		ema=True

# Train ARC-AGI-2 (single GPU, memory-optimized)
train-arc2:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/arc2concept-aug-1000]" \
		global_batch_size=32 \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=4 \
		arch.halt_max_steps=8 \
		+run_name=pretrain_att_arc2concept_1gpu \
		ema=True

# Train Sudoku-Extreme with MLP (single GPU, memory-optimized)
train-sudoku-mlp:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000]" \
		evaluators="[]" \
		global_batch_size=64 \
		epochs=100 eval_interval=5 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.mlp_t=True arch.pos_encodings=none \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=6 \
		arch.halt_max_steps=8 \
		+run_name=pretrain_mlp_t_sudoku_1gpu \
		ema=True

# Train Sudoku-Extreme with Attention (single GPU, memory-optimized)
train-sudoku:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000]" \
		evaluators="[]" \
		global_batch_size=64 \
		epochs=50000 eval_interval=5000 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=6 \
		arch.halt_max_steps=8 \
		+run_name=pretrain_att_sudoku_1gpu \
		ema=True

# Train Maze-Hard (single GPU, memory-optimized)
train-maze:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/maze-30x30-hard-1k]" \
		evaluators="[]" \
		global_batch_size=64 \
		epochs=50000 eval_interval=5000 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.L_layers=2 \
		arch.H_cycles=3 arch.L_cycles=4 \
		arch.halt_max_steps=8 \
		+run_name=pretrain_att_maze30x30_1gpu \
		ema=True

# ========== Memory Tuning Variants ==========
# If you still run out of memory, try these more aggressive settings:

# Ultra-low memory ARC training (batch size 16)
train-arc1-low-mem:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/arc1concept-aug-1000]" \
		global_batch_size=16 \
		arch.L_layers=1 \
		arch.H_cycles=2 arch.L_cycles=4 \
		arch.halt_max_steps=4 \
		arch.hidden_size=384 \
		+run_name=pretrain_att_arc1concept_1gpu_lowmem \
		ema=False

# Ultra-low memory Sudoku training (batch size 32)
train-sudoku-low-mem:
	python pretrain.py \
		arch=trm \
		data_paths="[$(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000]" \
		evaluators="[]" \
		global_batch_size=32 \
		epochs=50000 eval_interval=5000 \
		lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
		arch.L_layers=1 \
		arch.H_cycles=2 arch.L_cycles=4 \
		arch.halt_max_steps=4 \
		arch.hidden_size=384 \
		+run_name=pretrain_att_sudoku_1gpu_lowmem \
		ema=False
