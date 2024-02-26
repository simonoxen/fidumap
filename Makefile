# Directories
REPO_DIR=/mnt/arbeit/simon/repo/myKeymorph
MODELS_DIR=$(REPO_DIR)/models
RUNS_DIR=$(REPO_DIR)/runs
PYTHON_BIN_DIR=/home/ningfei/miniforge3/envs/torch_test/bin

# Executables
PYTHON_EXE=$(PYTHON_BIN_DIR)/python

# Python scripts
TRAIN_SCRIPT=$(REPO_DIR)/train.py
PRETRAIN_SCRIPT=$(REPO_DIR)/pretrain.py

# Params
N_KEYPOINTS=32

# Processed data
PRETRAIN_STATE=$(MODELS_DIR)/pretrain_state_$(N_KEYPOINTS)
TRAIN_STATE_INIT=$(MODELS_DIR)/train_state_init_$(N_KEYPOINTS)
TRAIN_STATE=$(MODELS_DIR)/train_state_$(N_KEYPOINTS)

.PHONY : train
train : $(TRAIN_STATE)

$(TRAIN_STATE) : $(TRAIN_STATE_INIT)
	$(PYTHON_EXE) $(TRAIN_SCRIPT) --n_epochs 2000 --n_keypoints $(N_KEYPOINTS) --load $< --save $@

$(TRAIN_STATE_INIT) : $(PRETRAIN_STATE)
	$(PYTHON_EXE) $(TRAIN_SCRIPT) --same_moving_fixed --n_epochs 500 --n_keypoints $(N_KEYPOINTS) --load $< --save $@

$(PRETRAIN_STATE) :
	$(PYTHON_EXE) $(PRETRAIN_SCRIPT) --n_epochs 500 --n_keypoints $(N_KEYPOINTS) --save $@

.PHONY : clear_models
clear_models : 
	@rm -fr $(MODELS_DIR)/*

.PHONY : clear_runs
clear_runs : 
	@rm -fr $(RUNS_DIR)/*