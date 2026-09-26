PYTHON := .venv/bin/python
CHECKPOINT := checkpoints/multistep_model.npz
OUT := output

.DEFAULT_GOAL := all
.PHONY: all train clean check-deps

all: check-deps train \
     $(OUT)/original_distribution.png \
     $(OUT)/reverse_evolution.mp4 \
     $(OUT)/learned_curve_on_joint.png \
     $(OUT)/learned_curve_on_joint_adjacent.png \
     $(OUT)/multistep_from_checkpoint.png \
     $(OUT)/step0_backward.png \
     $(OUT)/step0_forward_backward_beta0.3.png \
     $(OUT)/step0_x1_given_x0.png \
     $(OUT)/trajectory_image.png \
     $(OUT)/single_trajectory_landscape.png \
     $(OUT)/sample_grid.png

train: $(CHECKPOINT)

check-deps:
	@$(PYTHON) -c "import tkinter" >/dev/null 2>&1 || { \
		echo "ERROR: tkinter is not available for $(PYTHON)."; \
		echo "  Required by the interactive viewer reverse_process_viewer.py to open a display window."; \
		echo "  Install it via your OS package manager (e.g. 'sudo dnf install python3-tkinter' on Fedora), then recreate the venv."; \
		exit 1; \
	}
	@command -v ffmpeg >/dev/null 2>&1 || \
		echo "WARNING: ffmpeg not found on PATH -- reverse_evolution_video.py (output/reverse_evolution.mp4) will fail."

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

$(OUT)/step0_backward.png: step0_backward.py $(CHECKPOINT) ddpm_step.py hue_gmm.py train_denoiser.py mlp.py multistep_model.py schedule.py
	$(PYTHON) step0_backward.py

$(OUT)/step0_forward_backward_beta0.3.png: step0_forward_backward_large_noise.py ddpm_step.py hue_gmm.py
	$(PYTHON) step0_forward_backward_large_noise.py

$(OUT)/step0_x1_given_x0.png: step0_x1_given_x0.py ddpm_step.py hue_gmm.py
	$(PYTHON) step0_x1_given_x0.py

$(OUT)/trajectory_image.png: trajectory_image.py $(CHECKPOINT) multistep_model.py hue_gmm.py
	$(PYTHON) trajectory_image.py

$(OUT)/single_trajectory_landscape.png: single_trajectory_landscape.py $(CHECKPOINT) ddpm_step.py hue_gmm.py multistep_model.py schedule.py
	$(PYTHON) single_trajectory_landscape.py

$(OUT)/sample_grid.png: sample_viewer.py hue_gmm.py
	$(PYTHON) sample_viewer.py

clean:
	rm -rf $(OUT) $(CHECKPOINT)
