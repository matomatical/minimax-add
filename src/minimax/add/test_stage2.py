"""Incremental integration tests for Stage 2 (ADD w/o guidance).

Run with: tpu-device 0 python -m minimax.add.test_stage2
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents
from minimax.runners.eval_runner import EvalRunner

from minimax.add.runner import ADDRunner
from minimax.add.theta import decode_level, GRID_SIZE


def make_env_kwargs():
    return dict(
        height=13, width=13, n_walls=25, see_through_walls=True,
        agent_view_size=5, max_episode_steps=250, normalize_obs=False,
        sample_n_walls=True, replace_wall_pos=True,
    )


def make_runner_kwargs():
    env_kwargs = make_env_kwargs()
    dummy_env, _ = envs.make("Maze", env_kwargs=env_kwargs)
    n_actions = dummy_env.action_space().n
    student_model = models.make(
        env_name="Maze", model_name="default_student_cnn",
        output_dim=n_actions, recurrent_arch="lstm",
    )
    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=5, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    return dict(
        env_name="Maze", env_kwargs=env_kwargs,
        student_agents=[student_agent], n_students=1,
        n_parallel=32, n_eval=1, n_rollout_steps=256,
        lr=1e-4, discount=0.995, gae_lambda=0.95,
        track_env_metrics=False,
    )


def test_a_instantiation():
    print("=" * 60)
    print("TEST A: ADDRunner instantiation")
    print("=" * 60)
    runner_kwargs = make_runner_kwargs()
    t0 = time.time()
    runner = ADDRunner(
        diffusion_ckpt_path="checkpoints/diffusion/final.pkl",
        ddim_steps=50,
        **runner_kwargs,
    )
    dt = time.time() - t0
    print(f"  ADDRunner created in {dt:.1f}s")
    print(f"  Diffusion params: {jax.tree.map(lambda x: x.shape, runner.diff_params)['params'].keys()}")
    print(f"  use_guidance: {runner.use_guidance}")
    print(f"  n_parallel: {runner.n_parallel}, n_eval: {runner.n_eval}")
    assert not runner.use_guidance, "Should be unguided for Stage 2"
    print("  PASSED")
    return runner


def test_b_ddim_sampling(runner):
    print()
    print("=" * 60)
    print("TEST B: DDIM sampling + decode quality")
    print("=" * 60)
    rng = jax.random.PRNGKey(42)

    # First call triggers JIT compilation
    print("  Compiling DDIM (first call, may take minutes)...")
    t0 = time.time()
    thetas = runner._sample_thetas(rng, 32)
    thetas.block_until_ready()
    dt_compile = time.time() - t0
    print(f"  First batch (incl. compile): {dt_compile:.1f}s")

    # Second call is fast
    rng2 = jax.random.PRNGKey(123)
    t0 = time.time()
    thetas2 = runner._sample_thetas(rng2, 32)
    thetas2.block_until_ready()
    dt_fast = time.time() - t0
    print(f"  Second batch (cached): {dt_fast:.2f}s")
    print(f"  Thetas shape: {thetas.shape}, range: [{float(thetas.min()):.3f}, {float(thetas.max()):.3f}]")

    # Decode and check quality
    wall_maps, agent_pos, goal_pos, agent_dirs = jax.vmap(decode_level)(thetas)
    wall_maps_np = np.array(wall_maps)
    agent_pos_np = np.array(agent_pos)
    goal_pos_np = np.array(goal_pos)

    # Border walls
    border_ok = 0
    for i in range(32):
        wm = wall_maps_np[i]
        if wm[0, :].all() and wm[-1, :].all() and wm[:, 0].all() and wm[:, -1].all():
            border_ok += 1
    print(f"  Border walls intact: {border_ok}/32")
    assert border_ok == 32, "Border wall enforcement should make all borders intact"

    # Interior wall counts
    interior_walls = []
    for i in range(32):
        wm = wall_maps_np[i]
        border_count = 2 * GRID_SIZE + 2 * (GRID_SIZE - 2)  # 48
        total = int(wm.sum())
        interior_walls.append(total - border_count)
    iw = np.array(interior_walls)
    print(f"  Interior walls: {iw.mean():.1f} +/- {iw.std():.1f} (target ~28 +/- 16)")

    # Agent/goal overlap
    overlaps = 0
    for i in range(32):
        if np.array_equal(agent_pos_np[i], goal_pos_np[i]):
            overlaps += 1
    print(f"  Agent/goal overlaps: {overlaps}/32")

    # Agent/goal on walls
    agent_on_wall = 0
    goal_on_wall = 0
    for i in range(32):
        ar, ac = agent_pos_np[i]
        gr, gc = goal_pos_np[i]
        if wall_maps_np[i, ar, ac]:
            agent_on_wall += 1
        if wall_maps_np[i, gr, gc]:
            goal_on_wall += 1
    print(f"  Agent on wall: {agent_on_wall}/32, Goal on wall: {goal_on_wall}/32")

    print("  PASSED")
    return thetas


def test_c_set_env_instance(runner, thetas):
    print()
    print("=" * 60)
    print("TEST C: set_env_instance integration")
    print("=" * 60)

    instances = runner._decode_to_instances(thetas[:1])  # just one level
    print(f"  EnvInstance fields:")
    print(f"    agent_pos: {instances.agent_pos.shape}, dtype={instances.agent_pos.dtype}")
    print(f"    goal_pos: {instances.goal_pos.shape}, dtype={instances.goal_pos.dtype}")
    print(f"    agent_dir_idx: {instances.agent_dir_idx.shape}, dtype={instances.agent_dir_idx.dtype}")
    print(f"    wall_map: {instances.wall_map.shape}, dtype={instances.wall_map.dtype}")

    # Call set_env_instance through the wrapper chain
    single_instance = jax.tree.map(lambda x: x[0], instances)
    obs, state, extra = runner.benv.env.set_env_instance(single_instance)

    print(f"  obs type: {type(obs).__name__}, keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"    obs['{k}']: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  obs shape: {obs.shape}")
    print(f"  state.agent_pos: {state.agent_pos} (expect x,y = col,row)")
    print(f"  state.goal_pos: {state.goal_pos}")
    print(f"  state.wall_map shape: {state.wall_map.shape}")
    print(f"  extra keys: {list(extra.keys()) if isinstance(extra, dict) else type(extra)}")

    # Verify coordinate match
    decoded_wall_map, decoded_agent_rc, _, _ = decode_level(thetas[0])
    expected_xy = jnp.array([decoded_agent_rc[1], decoded_agent_rc[0]], dtype=jnp.uint32)
    actual_xy = state.agent_pos
    match = jnp.array_equal(expected_xy, actual_xy)
    print(f"  Coordinate check: decoded (r,c)={np.array(decoded_agent_rc)} "
          f"-> expected (x,y)={np.array(expected_xy)}, got {np.array(actual_xy)}, match={bool(match)}")
    assert match, "Coordinate conversion mismatch!"

    print("  PASSED")


def test_d_single_run(runner):
    print()
    print("=" * 60)
    print("TEST D: Single run() call")
    print("=" * 60)

    rng = jax.random.PRNGKey(0)
    print("  Calling runner.reset()...")
    t0 = time.time()
    runner_state = runner.reset(rng)
    dt = time.time() - t0
    print(f"  reset() done in {dt:.1f}s")

    print("  Calling runner.run() (first call compiles everything)...")
    t0 = time.time()
    result = runner.run(*runner_state)
    stats = result[0]
    # Block until computation is done
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, stats)
    dt = time.time() - t0
    print(f"  run() done in {dt:.1f}s (includes compilation)")

    print(f"  Stats keys: {sorted(stats.keys())}")
    for k, v in sorted(stats.items()):
        if not k.startswith("_"):
            print(f"    {k}: {float(v):.4f}")

    if "_thetas" in stats:
        thetas = stats["_thetas"]
        print(f"  _thetas shape: {thetas.shape}")
        assert thetas.shape == (32, 16, 16, 3), f"Expected (32, 16, 16, 3), got {thetas.shape}"

    # Check for NaNs
    nan_found = False
    for k, v in stats.items():
        if k.startswith("_"):
            continue
        if jnp.isnan(v).any():
            print(f"  WARNING: NaN in stats['{k}']")
            nan_found = True
    if not nan_found:
        print("  No NaN values in stats")

    # Second run should be fast
    runner_state = result[1:]
    t0 = time.time()
    result2 = runner.run(*runner_state)
    stats2 = result2[0]
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, stats2)
    dt2 = time.time() - t0
    print(f"  Second run(): {dt2:.2f}s")

    print("  PASSED")


def main():
    print(f"JAX devices: {jax.devices()}")
    print()

    runner = test_a_instantiation()
    thetas = test_b_ddim_sampling(runner)
    test_c_set_env_instance(runner, thetas)
    test_d_single_run(runner)

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
