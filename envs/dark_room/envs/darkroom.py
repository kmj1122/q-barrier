import warnings
from typing import Optional
import os
import time
import multiprocessing

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import uuid

# Optional import for DarkRoomGifWrapper (only needed for visualization)
try:
    from amago.envs.amago_env import AMAGOEnv
    _AMAGO_AVAILABLE = True
except ImportError:
    AMAGOEnv = object  # Fallback base class
    _AMAGO_AVAILABLE = False

# gym warnings are annoying
# warnings.filterwarnings("ignore")


class DarkRoom(gym.Env):
    metadata = {"render_mode": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        time_limit,
        size,
        goal_pos=None,
        start_pos=None,
        render_mode=None,
        terminate_on_goal=False,
        max_episodes=None,
    ):
        self.size = size
        self.time_limit = time_limit
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array((0, 0), dtype=np.int32),   # noop
            1: np.array((-1, 0), dtype=np.int32),  # up
            2: np.array((0, 1), dtype=np.int32),   # right
            3: np.array((1, 0), dtype=np.int32),   # down
            4: np.array((0, -1), dtype=np.int32),  # left
        }
        self.center_pos = (self.size // 2, self.size // 2)
        if start_pos is not None:
            self.start_pos = np.asarray(start_pos)
        else:
            self.start_pos = self.center_pos
        if goal_pos is not None:
            self.goal_pos = np.asarray(goal_pos)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal_pos()

        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode
        self.max_episodes = max_episodes if max_episodes else np.inf

    def generate_goal_pos(self):
        while True:
            goal_pos = self.np_random.integers(0, self.size, size=2)
            if np.array_equal(goal_pos, self.center_pos):
                continue
            return goal_pos

    def soft_reset(self):
        self.agent_pos = np.array(self.start_pos, dtype=np.int32)
        self.time_count = 0
        return self.agent_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_pos = np.array(self.start_pos, dtype=np.int32)
        self.episode_count = 0
        self.time_count = 0
        
        return self.agent_pos, {'soft_reset': 0}

    def step(self, action):
        """
        Time limit or potentially reaching the goal end the inner episodes. The outer episode is only terminated when a specific number of inner episodes are completed.
        """
        self.agent_pos = np.clip(self.agent_pos + self.action_to_direction[action], 0, self.size - 1)

        reward = .1 if np.array_equal(self.agent_pos, self.goal_pos) else 0.0

        info = {'soft_reset': 0}
        self.time_count += 1
        if self.time_count >= self.time_limit or (reward and self.terminate_on_goal):
            self.episode_count += 1
            # this will also trigger on the last episode of a trial. however this is consistent with the fact that the soft_reset flag 
            # is always on the first timestep of the new episode.
            self.soft_reset() 
            info['soft_reset'] = 1

        terminated = False
        if self.episode_count >= self.max_episodes:
            terminated = True

        return self.agent_pos, reward, terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full((self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8)
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid
        
        
class CostWrapper(gym.Wrapper):
    def __init__(
        self,
        env: DarkRoom,
        num,
        cost,
        seed,
        min_cost_to_go,
        max_cost_to_go,
        guranatee_path=False,
        reset_to_new_cost=False,
        radial_alpha=None,
        exclude_goal_and_start_from_cost=False,
        reset_budget_on_inner_episode=None,
    ):
        super().__init__(env)
        self.num = num
        self.cost_value = cost
        self.seed = seed
        self.guranatee_path = guranatee_path
        self.min_initial_cost_to_go = min_cost_to_go
        self.max_initial_cost_to_go = max_cost_to_go
        self.initial_cost_to_go = None
        self.current_cost_to_go = None
        self.reset_to_new_cost = reset_to_new_cost
        self.radial_alpha = radial_alpha
        self.exclude_goal_and_start_from_cost = bool(exclude_goal_and_start_from_cost)
        # The inner-episode boundary is detected inside DarkRoom.step(), which calls its OWN soft_reset()
        # and so never reaches CostWrapper.soft_reset() -- the budget therefore decays across the whole
        # 50-episode meta-episode instead of being restored per inner episode. Restoring it changes the
        # environment the released checkpoints were TRAINED in, so it is opt-in and OFF by default; the
        # env var lets a single eval enable it without touching config plumbing or in-flight training.
        if reset_budget_on_inner_episode is None:
            reset_budget_on_inner_episode = os.environ.get(
                "DARKROOM_RESET_BUDGET_PER_INNER_EPISODE", "0"
            ) == "1"
        self.reset_budget_on_inner_episode = bool(reset_budget_on_inner_episode)
        self._darkroom_env = getattr(self.env, "unwrapped", self.env)
        if not all(
            hasattr(self._darkroom_env, attr)
            for attr in ("size", "center_pos", "goal_pos", "start_pos", "soft_reset")
        ):
            raise AttributeError(
                "CostWrapper expects a DarkRoom-like base env with size/center_pos/goal_pos/start_pos/soft_reset."
            )
        
        # Generate cost function
        self.rng = np.random.default_rng(self.seed)
        # self._build_cost_function() # must be called in reset
        self.high_cost_pos_set = set()
        self._map_built = False
        self._grid_size = int(self._darkroom_env.size)
        self._all_positions = [
            (i, j) for i in range(self._grid_size) for j in range(self._grid_size)
        ]
        self._num_total_positions = len(self._all_positions)
        self._high_cost_mask = np.zeros((self._grid_size, self._grid_size), dtype=bool)
        self._radial_probs = None
        if self.radial_alpha:
            positions_arr = np.asarray(self._all_positions, dtype=np.float32)
            center = np.asarray(self._darkroom_env.center_pos, dtype=np.float32)
            probs = np.linalg.norm(positions_arr - center, axis=1)
            probs = np.exp(self.radial_alpha * probs)
            self._radial_probs = probs / probs.sum()
        
        # Modify observation space if cost_to_go is provided
        self.observation_space = spaces.Dict({
            'observation': env.observation_space,
            'cost_to_go': spaces.Box(low=0.0, high=float(self.max_initial_cost_to_go), shape=(), dtype=np.float32)
        })

    def _build_cost_function(self):
        """Build cost function with high-cost positions"""
        grid_size = self._grid_size
        num_to_sample = min(self.num, self._num_total_positions)
        
        if not self.radial_alpha:
            high_cost_positions = self.rng.choice(
                self._num_total_positions, size=num_to_sample, replace=False
            )
            self.high_cost_pos_set = {
                self._all_positions[i] for i in high_cost_positions.tolist()
            }
        else:
            sampled_positions = self.rng.choice(
                self._all_positions,
                size=num_to_sample,
                replace=False,
                p=self._radial_probs,
            ).tolist()
            self.high_cost_pos_set = {tuple(pos) for pos in sampled_positions}
    

        # Apply guarantee if needed
        if self.guranatee_path:
            self._ensure_path_guarantee(grid_size, self.initial_cost_to_go)
        
        if self.exclude_goal_and_start_from_cost:
            goal = tuple(np.asarray(self._darkroom_env.goal_pos, dtype=np.int32).tolist())
            start = tuple(np.asarray(self._darkroom_env.start_pos, dtype=np.int32).tolist())
            self.high_cost_pos_set.discard(goal)
            self.high_cost_pos_set.discard(start)

        self._high_cost_mask.fill(False)
        if self.high_cost_pos_set:
            xs, ys = zip(*self.high_cost_pos_set)
            self._high_cost_mask[np.asarray(xs, dtype=np.intp), np.asarray(ys, dtype=np.intp)] = True

    def _ensure_path_guarantee(self, grid_size, guarantee_below_cost):
        """Ensure path cost is below guarantee by removing obstacles"""
        center_pos = (grid_size // 2, grid_size // 2)
        goal_pos = tuple(np.asarray(self._darkroom_env.goal_pos, dtype=np.int32).tolist())
        
        def dijkstra_grid_with_path(start, goal, high_cost_set):
            import heapq
            pq = [(0, start)]
            distances = {start: 0}
            parents = {start: None}
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            while pq:
                current_cost, (x, y) = heapq.heappop(pq)
                if (x, y) == goal:
                    path = []
                    current = (x, y)
                    while current is not None:
                        path.append(current)
                        current = parents[current]
                    return current_cost, path[::-1]
                    
                if current_cost > distances.get((x, y), np.inf):
                    continue
                    
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        edge_cost = self.cost_value if (nx, ny) in high_cost_set else 0
                        new_cost = current_cost + edge_cost
                        if new_cost < distances.get((nx, ny), np.inf):
                            distances[(nx, ny)] = new_cost
                            parents[(nx, ny)] = (x, y)
                            heapq.heappush(pq, (new_cost, (nx, ny)))
            return np.inf, []
        
        path_cost, min_path = dijkstra_grid_with_path(center_pos, goal_pos, self.high_cost_pos_set)
        if path_cost > guarantee_below_cost and min_path:
            path_obstacles = [pos for pos in min_path[1:] if pos in self.high_cost_pos_set]
            if path_obstacles:
                excess_cost = path_cost - guarantee_below_cost
                obstacles_to_remove = min(len(path_obstacles), int(np.ceil(excess_cost / self.cost_value)))
                obstacles_to_remove_list = self.rng.choice(path_obstacles, size=obstacles_to_remove, replace=False)
                for obstacle in obstacles_to_remove_list:
                    self.high_cost_pos_set.remove(tuple(obstacle))

    def calculate_cost(self, obs):
        """Calculate cost for a given observation (x,y coordinates)"""
        x = int(obs[0])
        y = int(obs[1])
        if self._high_cost_mask[x, y]:
            return self.cost_value
        return 0.0
    
    def soft_reset(self):
        obs = self._darkroom_env.soft_reset()
        self.current_cost_to_go = self.initial_cost_to_go
        return {'observation': obs, 'cost_to_go': np.float32(self.current_cost_to_go)}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset cost budget if provided
        self.initial_cost_to_go = self.rng.uniform(self.min_initial_cost_to_go, self.max_initial_cost_to_go)
        self.current_cost_to_go = self.initial_cost_to_go

        if self.reset_to_new_cost or not self._map_built:
            self._build_cost_function()
            self._map_built = True

        cost = self.calculate_cost(obs)
        info['cost'] = cost
        info['cost_limit'] = np.float32(self.initial_cost_to_go)
        obs = {'observation': obs, 'cost_to_go': np.float32(self.current_cost_to_go)}
            
        return obs, info
    
    def step(self, action):
        # Extract state from dict observation
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        cost = self.calculate_cost(obs)
        info['cost'] = cost
        self.current_cost_to_go -= cost
        # DarkRoom.step() has already soft-reset the base env on this step, so `obs` is the first
        # observation of the NEW inner episode; restore the budget to match CostWrapper.soft_reset().
        if self.reset_budget_on_inner_episode and info.get('soft_reset', 0):
            self.current_cost_to_go = self.initial_cost_to_go
        info['cost_limit'] = np.float32(self.initial_cost_to_go)
        obs = {'observation': obs, 'cost_to_go': np.float32(self.current_cost_to_go)}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Print grid to console showing cost cells (x), target (t), and agent (o)"""
        grid_size = self._grid_size
        
        # Create empty grid
        grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Mark high cost positions with 'x'
        for pos in self.high_cost_pos_set:
            x, y = pos
            grid[x][y] = 'x'
        
        # Mark goal position with 't'
        goal_x, goal_y = self._darkroom_env.goal_pos
        grid[goal_x][goal_y] = 't'
        
        # Mark agent position with 'o' (agent overrides other symbols)
        agent_x, agent_y = self._darkroom_env.agent_pos
        grid[agent_x][agent_y] = 'o'
        
        # Print the grid
        print("\nGrid:")
        for row in grid:
            print(' '.join(row))
        print(f"Agent at ({agent_x}, {agent_y}), Goal at ({goal_x}, {goal_y})")
        print(f"Cost to go: {self.current_cost_to_go:.1f}")
        print()

class DarkRoomGifWrapper(AMAGOEnv):
    """
    A wrapper for DarkRoom environments that handles GIF creation during validation.
    
    Only creates GIFs when enabled=True (should only be set for environment 0).
    Uses a closure-based experiment provider to get current epoch information.
    
    Note: Requires 'amago' package for full functionality.
    """
    
    def __init__(
        self,
        env,
        env_name=None,
        get_experiment=None,
        enabled=False,
        save_dir=".",
        run_name="experiment",
        add_process_handle=None,
        name_postfix=None,
        **kwargs
    ):
        if not _AMAGO_AVAILABLE:
            raise ImportError(
                "DarkRoomGifWrapper requires 'amago' package. "
                "Install with: pip install amago"
            )
        super().__init__(env, env_name, **kwargs)
        self.get_experiment = get_experiment
        self.enabled = enabled
        self.save_dir = save_dir
        self.run_name = run_name
        self.gif_saved = -1
        self.gif_process = None
        self.add_process_handle = add_process_handle
        self.post_fix = name_postfix
        # GIF tracking state
        self.traj_states = []
        self.traj_actions = []
        self.traj_high_cost_pos_set = None
        self.traj_initial_cost_to_go = None
        self._gif_processes = []

    def _find_inner_env_with_attr(self, attr_name: str):
        env = self.env
        while env is not None:
            if hasattr(env, attr_name):
                return env
            env = getattr(env, "env", None)
        return None

    def _cost_env(self):
        env = self._find_inner_env_with_attr("high_cost_pos_set")
        if env is None:
            raise AttributeError("Could not locate CostWrapper in environment wrapper stack.")
        return env
        
    @property
    def current_epoch(self) -> int:
        """Get current epoch from experiment via closure"""
        if self.get_experiment:
            experiment = self.get_experiment()
            return getattr(experiment, "epoch", 0) if experiment else 0
        return 0
        
    def _should_create_gif(self) -> bool:
        """Check if we should create a GIF for this episode"""
        return (
            self.enabled and 
            self.gif_saved != self.current_epoch  # Only once per epoch
        )
        
    def _start_trajectory_tracking(self, timestep):
        """Start tracking trajectory for GIF creation"""
        if self._should_create_gif():
            self.traj_states = []
            self.traj_actions = []
            # observation IS the state for DarkRoom
            current_state = timestep.obs["observation"]
            if hasattr(current_state, 'cpu'):
                current_state = current_state.cpu().numpy()
            # Handle batch dimension - extract first element for single env
            if current_state.ndim > 1 and current_state.shape[0] == 1:
                current_state = current_state[0]
            self.traj_states.append(current_state)
            cost_env = self._cost_env()
            self.traj_high_cost_pos_set = set(cost_env.high_cost_pos_set)
            self.traj_initial_cost_to_go = cost_env.initial_cost_to_go
            
    def _track_action(self, action):
        """Track action for GIF creation"""
        if self._should_create_gif():
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            if hasattr(action, 'item'):
                action = action.item()
            self.traj_actions.append(action)
            
    def _track_next_state(self, timestep):
        """Track next state for GIF creation"""
        if self._should_create_gif():
            current_state = timestep.obs["observation"]
            if hasattr(current_state, 'cpu'):
                current_state = current_state.cpu().numpy()
            # Handle batch dimension - extract first element for single env
            if current_state.ndim > 1 and current_state.shape[0] == 1:
                current_state = current_state[0]
            self.traj_states.append(current_state)
            
    def _create_gif_async(self, states_array, actions_array):
        """Create GIF asynchronously using multiprocessing"""
        try:
            from utils.visualization import animate_traj
            
            cost_env = self._cost_env()
            darkroom_env = getattr(cost_env, "_darkroom_env", getattr(cost_env, "unwrapped", cost_env))
            goal_pos = darkroom_env.goal_pos
            env_size = darkroom_env.size
            
            # Get cost information from trajectory tracking
            high_cost_positions = set(self.traj_high_cost_pos_set) if self.traj_high_cost_pos_set is not None else None
            initial_cost_to_go = self.traj_initial_cost_to_go
            
            # Setup save directory
            logs_dir = os.path.join(self.save_dir, self.run_name, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            # gif_name = f"test_{time.strftime('%H%M')}_{uuid.uuid4()}"
            gif_name = f"epoch_{self.current_epoch}{'_' + self.post_fix if self.post_fix else ''}"
            
            # Create org_vis directory where GIF will be saved
            org_vis_dir = os.path.join(logs_dir, "org_vis")
            os.makedirs(org_vis_dir, exist_ok=True)
            
            # Start multiprocessing GIF creation
            self.gif_process = multiprocessing.Process(
                target=animate_traj,
                args=(env_size, states_array.copy(), actions_array.copy(), goal_pos),
                kwargs={
                    "name": gif_name,
                    "save_dir": org_vis_dir,
                    "high_cost_positions": high_cost_positions,
                    "initial_cost_to_go": initial_cost_to_go
                },
                daemon=False
            )
            self.gif_process.start()
            self._gif_processes.append(self.gif_process)
            if self.add_process_handle:
                self.add_process_handle(self.gif_process)
            self.gif_saved = self.current_epoch
            
        except Exception as e:
            print(f"Error creating GIF: {e}")

    def wait_for_gifs(self, timeout: float | None = None):
        """Wait for any outstanding GIF writer processes to finish."""
        remaining = []
        for process in self._gif_processes:
            if process is None:
                continue
            process.join(timeout=timeout)
            if process.is_alive():
                remaining.append(process)
        self._gif_processes = remaining
            
    def reset(self, seed=None, options=None):
        """Reset environment and start trajectory tracking if needed"""
        timestep, info = super().reset(seed=seed, options=options)
        if not self.enabled:
            return timestep, info
        self._start_trajectory_tracking(timestep)
        return timestep, info
        
    def step(self, action):
        """Step environment and handle GIF creation"""
        if not self.enabled:
            return super().step(action)

        # Track action before step
        should_create = self._should_create_gif()
        if should_create:
            self._track_action(action)
            
        # Take the step
        timestep, reward, terminated, truncated, info = super().step(action)
        done = bool(np.any(terminated) or np.any(truncated))
        
        # Track next state
        if should_create:
            self._track_next_state(timestep)
            
        # Create GIF if episode is done and we should
        if done and should_create and len(self.traj_states) > 1:
            states_array = np.array(self.traj_states[:-1])  # Exclude last state
            actions_array = np.array(self.traj_actions)
            
            if len(states_array) > 0 and len(actions_array) > 0:
                self._create_gif_async(states_array, actions_array)
                
        return timestep, reward, terminated, truncated, info
