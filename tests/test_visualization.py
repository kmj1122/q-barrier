import numpy as np
from PIL import Image

from utils.visualization import animate_traj


def test_darkroom_gif_uses_headless_pillow_writer(tmp_path):
    animate_traj(3, np.array([[1, 1], [1, 2]]), np.array([0, 2]),
                 goal=(2, 2), name="smoke", save_dir=tmp_path,
                 high_cost_positions=[(0, 0)], initial_cost_to_go=1.0)
    with Image.open(tmp_path / "basic_animation_smoke.gif") as image:
        assert image.format == "GIF"
        assert image.n_frames == 2
