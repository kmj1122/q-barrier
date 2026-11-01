from envs.dark_room.envs.darkroom import DarkRoom

# Register environments with gymnasium
from gymnasium import register as register_gymnasium

# Register DarkRoom environment
register_gymnasium(
    id="DarkRoomHard-v0",
    entry_point=lambda size=17, terminate_on_goal=True, **kwargs: DarkRoom(size=size, terminate_on_goal=terminate_on_goal, **kwargs),
)

# register safety-gymnasium environments
try:
    import safety_gymnasium
except:
    print("safety_gymnasium not found. Skipping registration of safety-gymnasium environments.")