# 3D Volume Registration via GA-Planes DVF

Non-rigid 3D image-space volume registration using Geometric Algebra (GA) Plane representation.

## Repository Layout
* `src/model.py`: Implementation of the multi-resolution GA-Planes network.
* `src/registration.py`: Backward-warping fields, analytical Jacobian determinant computation, and TV loss steps.
* `src/utils.py`: Logging pipelines and central slice extraction routines across multi-planar projections.
* `main.py`: Central training loop defining parameter constants and tracking workflows.

## Features
* **Image Space Tracking**: Directly optimizes Mean Squared Error (MSE) constraints across moving and target volume boundaries.
* **Regularization Mechanics**: Blends analytical 3x3 deformation Jacobian determinant constraints with direct field and grid Total Variation penalties.
* **Resolution Schedules**: Dynamically samples low-resolution proxy grids down-stream during early optimization iterations to bypass local minima.
* **Static Fallback Compatibility**: Preserves dynamic placeholder layouts internally to switch smoothly to hypervolume data loops if temporal parameters change.

## Execution
Ensure your volumes are saved in `data/moving_volume.pt` and `data/target_volume.pt` as `(1, 1, D, H, W)` tensors, then trigger the iteration run:
```bash
python main.py
