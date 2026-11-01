import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
from safety_gymnasium import make
from envs.mujoco.patch import patch_safety_gymnasium, VELOCITY_DEFAULT_THRESHOLDS
import logging
logger = logging.getLogger(__name__)
from utils.video_recorder import VideoRecorder
from PIL import Image, ImageDraw, ImageFont
from utils.timer import Timer

 


class MujocoWrapper(gym.Wrapper):
    """
    Wrapper for safety-gymnasium environments.
    Supports Goal and Velocity tasks with meta-learning:
    - Goal tasks: Car, Point agents with spatial navigation
    - Velocity tasks: Hopper, Ant, Humanoid, Walker2d, HalfCheetah with speed constraints
    - Meta-learning: Velocity thresholds vary per outer episode, constant within inner episodes
    """

    def __init__(self, env_name,
                 min_cost_to_go=0.0,
                 max_cost_to_go=100.0,
                 seed=None,
                 time_limit=1000,
                 max_episodes=None,
                 obs_no_goal=True,
                 radial_alpha=None,
                 task_type="goal",
                 normalize_reward=True,
                 sticky_movement_multiplier: int = 1,
                 lidar=False,
                 **make_kwargs
                 ):

        # Apply patches before environment creation
        patch_safety_gymnasium(radial_alpha, lidar, task_type)

        # Set up velocity threshold for velocity tasks
        if task_type == "velocity":
            # Get default threshold for this environment
            default_threshold = VELOCITY_DEFAULT_THRESHOLDS.get(env_name, 1.0)

            # Default: 50% to 200% of default (25% to 100% of max speed)
            self.min_velocity_threshold = default_threshold * 0.5
            self.max_velocity_threshold = default_threshold * 2.0

            # Will be sampled at reset time
            self.current_velocity_threshold = None

            # Calculate reward normalization factor for velocity tasks
            # Normalize by (max_steps * max_velocity / divisor)
            # Max velocity is 2x the default threshold (since default = 50% of max)
            max_velocity = default_threshold * 2.0
            self.reward_normalization_factor = time_limit * max_velocity / 5.0 if normalize_reward else 1.0
        else:
            self.current_velocity_threshold = None
            self.reward_normalization_factor = 1.0  # No normalization for goal tasks

        # When running on headless servers without an X display, force EGL backend
        # so that MuJoCo can create an OpenGL context without X/GLX.
        if not os.environ.get("DISPLAY"):
            os.environ.setdefault("MUJOCO_GL", "egl")

        # Create environment - pass velocity_threshold for velocity tasks if available
        # Note: threshold will be resampled at each outer reset
        if task_type == "velocity" and self.current_velocity_threshold is not None:
            env = make(
                env_name,
                max_episode_steps=float('inf'),
                velocity_threshold=self.current_velocity_threshold,
                **make_kwargs
            )
        else:
            env = make(
                env_name,
                max_episode_steps=float('inf'),
                **make_kwargs
            )
        # Wrap with sticky movement multiplier if requested
        if sticky_movement_multiplier is None:
            sticky_movement_multiplier = 1
        if int(sticky_movement_multiplier) > 1:
            env = StickyActionRepeatWrapper(env, multiplier=int(sticky_movement_multiplier))

        # Only set task attributes for goal tasks (velocity tasks don't have task attribute)
        if task_type == "goal":
            env.unwrapped.task.mechanism_conf.continue_goal = False
            env.unwrapped.task.num_steps = float('inf')

        super().__init__(env)
        self.task_type = task_type
        self.min_initial_cost_to_go = min_cost_to_go
        self.max_initial_cost_to_go = max_cost_to_go
        self.initial_cost_to_go = None
        self.current_cost_to_go = None
        self.time_limit = time_limit
        self.max_episodes = max_episodes
        # Set up RNG for cost_to_go sampling
        self.rng = np.random.default_rng(seed)
        
        # Modify observation space to include cost_to_go
        self.observation_space = spaces.Dict({
            'observation': env.observation_space,
            'cost_to_go': spaces.Box(
                low=0.0, 
                high=float(self.max_initial_cost_to_go), 
                shape=(), 
                dtype=np.float32
            )
        })

    def _soft_reset(self):
        """
        Soft reset for inner episodes.
        For velocity tasks: keeps the SAME velocity threshold (meta-learning requirement).
        """
        builder = self.env.unwrapped

        if self.task_type == "goal":
            # Goal tasks have task attribute
            task = builder.task
            # task.update_layout()
            # task.build_agent_position()
            # task.build_goal_position()
            task.world.set_state(self.state)
            task.update_world()
            task.steps = 0
            builder.terminated = False
            task.agent.reset()
            obs = task.obs()
        else:
            # Velocity tasks use standard MuJoCo state management
            builder.set_state(self.state[0], self.state[1])
            obs = builder._get_obs()

        self.current_cost_to_go = np.float32(self.initial_cost_to_go)
        self.step_count = 0

        return obs

    def _get_obs(self, env_obs):
        structured_obs = {
            'observation': env_obs,
            'cost_to_go': np.float32(self.current_cost_to_go)
        }
        return structured_obs
    
    def _get_info(self, soft_reset, cost, original_reward=None):
        info = {
            'soft_reset': soft_reset,
            'cost': cost,
            'cost_limit': np.float32(self.initial_cost_to_go)
        }
        # Add original (unnormalized) reward for logging
        if original_reward is not None:
            info['original_reward'] = np.float32(original_reward)
            # For wandb logging via AMAGO special metrics
            info['AMAGO_LOG_METRIC original_reward'] = np.float32(original_reward) #! useless, we need return, not just avg step reward
        # Add velocity threshold for velocity tasks
        if self.task_type == "velocity" and self.current_velocity_threshold is not None:
            # For test script metrics and general use
            info['velocity_threshold'] = np.float32(self.current_velocity_threshold)
            # For wandb logging via AMAGO special metrics
            info['AMAGO_LOG_METRIC velocity_threshold'] = np.float32(self.current_velocity_threshold)
        return info
    
    def reset(self, seed=None, options=None):
        """
        Returns observation consisting of cost_to_go and observation.
        For velocity tasks: samples a new velocity threshold per outer episode.
        """

        # Sample new velocity threshold for velocity tasks (outer episode reset)
        if self.task_type == "velocity":
            self.current_velocity_threshold = self.rng.uniform(
                self.min_velocity_threshold,
                self.max_velocity_threshold
            )
            # Update the environment's threshold
            self.env.unwrapped._velocity_threshold = self.current_velocity_threshold

        obs, _ = self.env.reset(seed=seed, options=options)

        # Store state for soft resets (different for goal vs velocity tasks)
        if self.task_type == "goal":
            self.state = self.env.unwrapped.task.world.get_state()
        else:
            # Velocity tasks use standard MuJoCo state (qpos, qvel)
            unwrapped = self.env.unwrapped
            self.state = (unwrapped.data.qpos.copy(), unwrapped.data.qvel.copy())

        self.initial_cost_to_go = self.rng.uniform(self.min_initial_cost_to_go, self.max_initial_cost_to_go)
        self.current_cost_to_go = self.initial_cost_to_go
        self.current_episode_count = 0
        self.step_count = 0

        structured_obs = self._get_obs(obs)
        info = self._get_info(0, 0)

        return structured_obs, info
    
    def step(self, action):
        """
        Step the mujoco environment and return structured observation with cost tracking.
        For velocity tasks, normalizes reward by (max_steps * max_velocity / divisor).
        """
        self.step_count += 1
        obs, reward, cost, terminated, truncated, _ = self.env.step(action)
        assert not truncated, "Truncated should not be set"

        self.current_cost_to_go -= cost

        # Normalize reward and store original for logging
        original_reward = reward
        normalized_reward = reward / self.reward_normalization_factor
        info = self._get_info(0, cost, original_reward=original_reward)

        if self.step_count >= self.time_limit or terminated:
            # Track violation before soft reset
            violated = 1.0 if self.current_cost_to_go < 0 else 0.0
            info['AMAGO_LOG_METRIC violation'] = np.float32(violated)

            obs = self._soft_reset()
            info['soft_reset'] = 1
            self.current_episode_count += 1
            if self.current_episode_count >= self.max_episodes:
                terminated = True
            else:
                terminated = False


        structured_obs = self._get_obs(obs)

        return structured_obs, normalized_reward, terminated, False, info

class StickyActionRepeatWrapper(gym.Wrapper):
    """
    Repeats the same action N times. Aggregates reward and cost across repeats,
    returns the last observation and a merged info. Terminates early if the
    underlying env signals terminated or truncated.
    
    Step signature matches Safety-Gymnasium's extended return:
        (obs, reward, cost, terminated, truncated, info)
    """

    def __init__(self, env, multiplier: int = 1):
        super().__init__(env)
        if multiplier is None:
            multiplier = 1
        if multiplier < 1:
            raise ValueError("sticky movement multiplier must be >= 1")
        self.multiplier = int(multiplier)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        repeats = self.multiplier
        total_reward = 0.0
        total_cost = 0.0
        terminated = False
        truncated = False
        last_obs = None

        for _ in range(repeats):
            obs, reward, cost, term, trunc, _ = self.env.step(action)
            total_reward += reward
            total_cost += cost
            terminated = terminated or term
            truncated = truncated or trunc
            last_obs = obs
            if term or trunc:
                break

        return last_obs, total_reward, total_cost, terminated, truncated, {}

class MujocoVideoRecorder(gym.Wrapper):
    def __init__(self, env,
                 output_dir, 
                 fps: int = 30,
                 enabled: bool = False,
                 epoch_shared=None,
                 run_name: str | None = None,
                 name_postfix: str | None = None,
                 ):
        super().__init__(env)
        self.enabled = enabled
        self.epoch_shared = epoch_shared
        self.base_output_dir = output_dir
        self.run_name = run_name
        self.fps = fps
        self._recorder: VideoRecorder | None = None
        self._last_recorded_epoch = -1
        self.name_postfix = name_postfix
        self._output_dir = self._build_output_dir()
        self._video_timer = Timer()

    def _find_attr_in_wrappers(self, attr_name, default=None):
        """
        Traverse the wrapper chain to find an attribute.
        Returns first found value or default.
        """
        try:
            ref = self.env
            visited = set()
            while ref is not None and id(ref) not in visited:
                visited.add(id(ref))
                if hasattr(ref, attr_name):
                    return getattr(ref, attr_name)
                # gym.Wrapper stores underlying env in `.env`, unwrapped base lacks it
                ref = getattr(ref, 'env', None)
        except Exception:
            pass
        return default

    def _build_overlay_text(self) -> str | None:
        """Build the overlay text: CTG and inner episode number if available."""
        ctg = self._find_attr_in_wrappers('current_cost_to_go', None)
        inner_ep = self._find_attr_in_wrappers('current_episode_count', None)
        if ctg is None and inner_ep is None:
            logger.warning("Video recorder: No CTG or inner episode count found in the environment")
            return None
        parts = []
        if ctg is not None:
            try:
                parts.append(f"CTG: {float(ctg):.2f}")
            except Exception:
                parts.append(f"CTG: {ctg}")
        if inner_ep is not None:
            try:
                # Display inner episode as 1-based for readability
                parts.append(f"ep: {int(inner_ep) + 1}")
            except Exception:
                parts.append(f"ep: {inner_ep}")
        return " | ".join(parts) if parts else None

    def _overlay_text_on_frame(self, frame):
        """
        Draw red text with a black outline at the top-right corner of the frame.
        Accepts and returns an RGB uint8 numpy array.
        """
        try:
            text = self._build_overlay_text()
            if not text:
                return frame

            # Convert to PIL Image for reliable RGB drawing
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            # Measure text size
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = draw.textsize(text, font=font)

            padding = 10
            x = max(padding, frame.shape[1] - text_w - padding)
            y = padding

            # Draw outline for contrast
            outline_color = (0, 0, 0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

            # Draw main text in red
            draw.text((x, y), text, font=font, fill=(255, 0, 0))

            return np.array(img, dtype=np.uint8)
        except Exception:
            # On any failure, return original frame to avoid breaking recording
            return frame

    def _build_output_dir(self):
        try:
            import os
            if self.run_name is None:
                raise ValueError("run_name must be provided for MujocoVideoRecorder")
            logs_dir = os.path.join(self.base_output_dir, self.run_name, "logs")
            videos_dir = os.path.join(logs_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)
            return videos_dir
        except Exception:
            # Fail fast if we cannot create directories
            raise

    @property
    def _current_epoch(self) -> int:
        return int(self.epoch_shared.value)

    def _should_record(self) -> bool:
        return self.enabled and self._last_recorded_epoch != self._current_epoch

    def _ensure_recorder(self):
        # Lazily create a recorder when we start a new epoch recording
        if self._recorder is None:
            # Standardize naming: epoch_{epoch_number}.mp4
            if self.name_postfix is None:
                video_filename = f"epoch_{self._current_epoch}"
            else:
                video_filename = f"test_{self.name_postfix}"
            self._recorder = VideoRecorder(
                output_dir=self._output_dir,
                video_name=video_filename,
                fps=self.fps,
                save_individual_frames=False,
            )
            self._video_timer.reset()

    def _record_frame(self):
        if not self.enabled:
            return
        try:
            frame = self.env.render()
            if self._recorder is not None:
                frame = self._overlay_text_on_frame(frame)
                self._recorder.add_frame(frame)
        except Exception as e: 
            logger.warning(f"Failed to record frame: {e}")

    def reset(self, seed=None, options=None):
        self._video_timer.start()
        try:
            obs, info = self.env.reset(seed=seed, options=options)
            # Start a fresh recording if applicable
            if self._should_record():
                # Close any previous recorder by saving it first
                self._save_video_if_any()
                self._ensure_recorder()
                # capture initial frame after reset
                self._record_frame()
            return obs, info
        finally:
            self._video_timer.pause()

    def step(self, action):
        self._video_timer.start()
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            # record the frame after the step (post-transition)
            if self._should_record():
                # If just transitioned to a new epoch mid-run, start a new recorder
                if self._recorder is None:
                    self._ensure_recorder()
                self._record_frame()

            done = terminated or truncated
            if done and self.enabled:
                # Save and mark epoch as recorded
                self._save_video_if_any()
                self._last_recorded_epoch = self._current_epoch
            return obs, reward, terminated, truncated, info
        finally:
            self._video_timer.pause()

    def _save_video_if_any(self):
        if self._recorder is not None and self._recorder.get_frame_count() > 0:
            self._video_timer.start()
            try:
                # path = self._recorder.save_mp4()
                path = self._recorder.save_gif()
                logger.info(f"Saved Mujoco video to: {path}")
                self._video_timer.pause()
                logger.info("Mujoco video total time: %.2f ms", self._video_timer.total_ms)
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")
            finally:
                # Reset recorder for next epoch/episode
                self._video_timer.reset()
                self._recorder = None

    def close(self):
        # Save any pending video on close
        if self.enabled:
            self._save_video_if_any()
        return self.env.close()
