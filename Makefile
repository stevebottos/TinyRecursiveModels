STORAGE_ROOT = /media/steve/storage

# ARC-AGI-1
arc1:
	python -m dataset.build_arc_dataset \
		--input-file-prefix kaggle/combined/arc-agi \
		--output-dir $(STORAGE_ROOT)/data/arc1concept-aug-1000 \
		--subsets training evaluation concept \
		--test-set-name evaluation

# ARC-AGI-2
arc2:
	python -m dataset.build_arc_dataset \
		--input-file-prefix kaggle/combined/arc-agi \
		--output-dir $(STORAGE_ROOT)/data/arc2concept-aug-1000 \
		--subsets training2 evaluation2 concept \
		--test-set-name evaluation2

# Sudoku-Extreme
sudoku:
	python dataset/build_sudoku_dataset.py \
		--output-dir $(STORAGE_ROOT)/data/sudoku-extreme-1k-aug-1000 \
		--subsample-size 1000 \
		--num-aug 1000

# Maze-Hard
maze:
	python dataset/build_maze_dataset.py \
		--output-dir $(STORAGE_ROOT)/data/maze-30x30-hard-1k

# Prepare all datasets
prepare-data: arc1 arc2 sudoku maze
