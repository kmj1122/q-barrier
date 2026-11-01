import numpy as np

def _patch_agent_start_location():
    from safety_gymnasium.bases.base_agent import BaseAgent

    base_init = BaseAgent.__init__

    def __init__(self, *args, **kwargs):
        base_init(self, *args, **kwargs) 
        self.locations = [(0.0, 0.0)]
        self.keepout = 0.1

    BaseAgent.__init__ = __init__

def _patch_object_placement(radial_alpha=None):
    from safety_gymnasium.utils.random_generator import RandomGenerator

    def draw_placement(self, placements: dict, keepout: float) -> np.ndarray:
        """Sample an (x,y) location, based on potential placement areas.

        Args:
            placements (dict): A list of (xmin, xmax, ymin, ymax) tuples that specify
              rectangles in the XY-plane where an object could be placed.
            keepout (float): Describes how much space an object is required to have
              around it, where that keepout space overlaps with the placement rectangle.

        Note:
            To sample an (x,y) pair, first randomly select which placement rectangle
            to sample from, where the probability of a rectangle is weighted by its
            area. If the rectangles are disjoint, there's an equal chance the (x,y)
            location will wind up anywhere in the placement space. If they overlap, then
            overlap areas are double-counted and will have higher density. This allows
            the user some flexibility in building placement distributions. Finally,
            randomly draw a uniform point within the selected rectangle.
        """
        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert constrained, 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.random_generator.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice

        # complete here:
        if radial_alpha is None or radial_alpha == 0:
            return np.array(
                [self.random_generator.uniform(xmin, xmax), self.random_generator.uniform(ymin, ymax)],
            )
        else:
            samples = self.random_generator.uniform(xmin, xmax, size=8192)
            scores = np.exp(radial_alpha * np.abs(samples - (xmin + xmax) / 2))
            # max_score = np.exp(radial_alpha * np.abs(xmin - (xmin + xmax) / 2)) if radial_alpha > 0 else 1
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            keep_mask = self.random_generator.uniform(0, 1, size=8192) < normalized_scores
            keep_samples = samples[keep_mask]
            return self.random_generator.choice(keep_samples, size=2)
    
    RandomGenerator.draw_placement = draw_placement

def _patch_goal_task():
    """
    Modify tasks such that they do not change goal position up on reaching the goal.
    """
    from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
    
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        # reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def update_world(self):
        self.last_dist_goal = self.dist_goal()

    GoalLevel0.update_world = update_world
    GoalLevel0.calculate_reward = calculate_reward

def _patch_obs(lidar=False):
    from safety_gymnasium.bases.base_task import BaseTask
    import mujoco   
    import gymnasium

    original_build_observation_space = BaseTask.build_observation_space
    def build_observation_space(self):
        original_build_observation_space(self)
        if not lidar:
            self.obs_info.obs_space_dict.spaces = {k: v for k, v in self.obs_info.obs_space_dict.spaces.items() if 'lidar' not in k}
        self.obs_info.obs_space_dict['agent_xy'] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float64,
        )
        self.obs_info.obs_space_dict['agent_mat'] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, 3),
            dtype=np.float64,
        )
        
        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict
        

    def obs(self) -> dict | np.ndarray:
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        if lidar:
            for obstacle in self._obstacles:
                if obstacle.is_lidar_observed:
                    obs[obstacle.name + '_lidar'] = self._obs_lidar(obstacle.pos, obstacle.group)
        
        obs['agent_xy'] = self.agent.pos[:2]
        obs['agent_mat'] = self.agent.mat

        assert self.obs_info.obs_space_dict.contains(
            obs,
        ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'


        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    BaseTask.obs = obs
    BaseTask.build_observation_space = build_observation_space


def _patch_velocity_threshold():
    """
    Patch velocity environments to allow dynamic threshold setting.
    This enables meta-learning by varying thresholds per outer episode.
    """
    # Import all v1 velocity environments
    from safety_gymnasium.tasks.safe_velocity.safety_hopper_velocity_v1 import SafetyHopperVelocityEnv
    from safety_gymnasium.tasks.safe_velocity.safety_ant_velocity_v1 import SafetyAntVelocityEnv
    from safety_gymnasium.tasks.safe_velocity.safety_humanoid_velocity_v1 import SafetyHumanoidVelocityEnv
    from safety_gymnasium.tasks.safe_velocity.safety_walker2d_velocity_v1 import SafetyWalker2dVelocityEnv
    from safety_gymnasium.tasks.safe_velocity.safety_half_cheetah_velocity_v1 import SafetyHalfCheetahVelocityEnv

    # Patch each environment class to accept custom threshold
    for env_class in [SafetyHopperVelocityEnv, SafetyAntVelocityEnv,
                      SafetyHumanoidVelocityEnv, SafetyWalker2dVelocityEnv,
                      SafetyHalfCheetahVelocityEnv]:
        original_init = env_class.__init__

        def make_patched_init(orig_init):
            def patched_init(self, velocity_threshold=None, **kwargs):
                # Call original init
                orig_init(self, **kwargs)
                # Override threshold if provided, otherwise keep default
                if velocity_threshold is not None:
                    self._velocity_threshold = velocity_threshold
            return patched_init

        env_class.__init__ = make_patched_init(original_init)


def patch_safety_gymnasium(radial_alpha=None, lidar=False, task_type="goal"):
    """Patch safety-gymnasium with custom implementations.

    Args:
        radial_alpha: Distribution parameter for spatial sampling (goal tasks only)
        lidar: Whether to add lidar observations
        task_type: "goal" or "velocity" - determines which patches to apply
    """
    _patch_agent_start_location()
    _patch_obs(lidar)

    if task_type == "goal":
        # Only apply spatial patches for goal tasks
        _patch_goal_task()
        _patch_object_placement(radial_alpha)
    elif task_type == "velocity":
        # Apply velocity-specific patches
        _patch_velocity_threshold()


# Export default thresholds for use in wrapper
VELOCITY_DEFAULT_THRESHOLDS = {
    'SafetyHopperVelocity-v1': 0.7402,
    'SafetyAntVelocity-v1': 2.6222,
    'SafetyHumanoidVelocity-v1': 1.4149,
    'SafetyWalker2dVelocity-v1': 2.3415,
    'SafetyHalfCheetahVelocity-v1': 3.2096,
}