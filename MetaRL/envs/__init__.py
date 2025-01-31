from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'HalfCheetahVel-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
    max_episode_steps=200
)


register(
    'HalfCheetahMassVel-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahMassVelEnv'},
    max_episode_steps=200
)
register(
    'Walker2dMassVel-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dMassVelEnv'},
    max_episode_steps=200
)
register(
    'Walker2dVel-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dVelEnv'},
    max_episode_steps=200
)
register(
    'ReacherPos-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.reacher:ReacherPosEnv'},
    max_episode_steps=50
)
