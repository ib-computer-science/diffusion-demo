PYTHON := .venv/bin/python
CHECKPOINT := checkpoints/multistep_model.npz
OUT := output

.DEFAULT_GOAL := all
.PHONY: all train clean

all: train \
     $(OUT)/original_distribution.png \
     $(OUT)/reverse_evolution.mp4 \
     $(OUT)/learned_curve_on_joint.png \
     $(OUT)/learned_curve_on_joint_adjacent.png \
     $(OUT)/multistep_from_checkpoint.png \
     $(OUT)/step0_forward_backward.png \
     $(OUT)/step0_forward_backward_beta0.3.png \
     $(OUT)/trajectory_image.png \
     $(OUT)/single_trajectory_landscape.png

train: $(CHECKPOINT)

$(OUT)/original_distribution.png: original_distribution.py hue_gmm.py
	$(PYTHON) original_distribution.py

$(CHECKPOINT): train_multistep_model.py multistep_model.py mlp.py hue_gmm.py schedule.py ddpm_step.py
	$(PYTHON) train_multistep_model.py

$(OUT)/reverse_evolution.mp4: reverse_evolution_video.py $(CHECKPOINT) multistep_model.py hue_gmm.py
	$(PYTHON) reverse_evolution_video.py

$(OUT)/learned_curve_on_joint.png: learned_curve_on_joint.py train_denoiser.py ddpm_step.py hue_gmm.py mlp.py
	$(PYTHON) learned_curve_on_joint.py

$(OUT)/learned_curve_on_joint_adjacent.png: learned_curve_on_joint_adjacent.py $(CHECKPOINT) ddpm_step.py hue_gmm.py multistep_model.py schedule.py
	$(PYTHON) learned_curve_on_joint_adjacent.py

$(OUT)/multistep_from_checkpoint.png: plot_multistep_from_checkpoint.py $(CHECKPOINT) multistep_model.py hue_gmm.py
	$(PYTHON) plot_multistep_from_checkpoint.py

$(OUT)/step0_forward_backward.png: step0_forward_backward.py ddpm_step.py hue_gmm.py
	$(PYTHON) step0_forward_backward.py

$(OUT)/step0_forward_backward_beta0.3.png: step0_forward_backward_large_noise.py step0_forward_backward.py ddpm_step.py hue_gmm.py
	$(PYTHON) step0_forward_backward_large_noise.py

$(OUT)/trajectory_image.png: trajectory_image.py $(CHECKPOINT) multistep_model.py hue_gmm.py
	$(PYTHON) trajectory_image.py

$(OUT)/single_trajectory_landscape.png: single_trajectory_landscape.py $(CHECKPOINT) ddpm_step.py hue_gmm.py multistep_model.py schedule.py
	$(PYTHON) single_trajectory_landscape.py

clean:
	rm -rf $(OUT) $(CHECKPOINT)
