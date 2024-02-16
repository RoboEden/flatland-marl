import argparse
import json
import os
import random
import time

import numpy as np
import torch
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.rail_generators import SparseRailGen
from flatland.envs.rail_generators import rail_from_grid_transition_map

from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import (
    SyncDataCollector,
    MultiSyncDataCollector,
    MultiaSyncDataCollector,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import ParallelEnv
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.modules import ActorValueOperator, ValueOperator
from torchrl.data import TensorDictReplayBuffer

from flatland_cutils import TreeObsForRailEnv as TreeCutils
from solution.nn.net_tree_torchrl import actor_net, critic_net, embedding_net
from solution.nn.net_tree_transformer import transformer_embedding_net
from flatland_torchrl.torchrl_rail_env import TorchRLRailEnv, TDRailEnv
from flatland_torchrl.custom_map_generator import generate_custom_rail


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--use-torch-deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--env-id", type=str, default="flatland-rl", help="the id of the environment"
    )
    parser.add_argument(
        "--num-agents", type=int, default=2, help="number of agents in the environment"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,  # for rollout
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=10, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs", type=int, default=4, help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--norm-adv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.01,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.2,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--pretrained-network-path",
        type=str,
        default=None,
        help="path to the pretrained network to be used",
    )
    parser.add_argument(
        "--delay-reward-coef",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--shortest-path-reward-coef",
        type=float,
        default=0,
    )
    parser.add_argument("--departure-reward-coef", type=float, default=0)
    parser.add_argument("--arrival-reward-coef", type=float, default=1)
    parser.add_argument("--deadlock-penalty-coef", type=float, default=0)
    parser.add_argument("--arrival-delay-penalty-coef", type=float, default=0)
    parser.add_argument("--map-width", type=int, default=30)
    parser.add_argument("--map-height", type=int, default=35)
    parser.add_argument("--curriculum-path", type=str, default=None)

    parser.add_argument(
        "--do-multisync", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-transformer-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--zero-tree-attributes", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--value-loss", type=str, default="smooth_l1")
    parser.add_argument("--map-name", type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    os.mkdir(f"model_checkpoints/{run_name}")

    if args.exp_name is not None:
        print("initializing tracking")

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.use_torch_deterministic
    print(f"using deterministic: {args.use_torch_deterministic}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"device used: {device}")

    # Set-up of the neural network

    if args.use_transformer_embedding:
        common_module = TensorDictModule(
            transformer_embedding_net(),
            in_keys=[("agents", "observation")],
            out_keys=[("hidden", "embedding"), ("hidden", "att_embedding")],
        )
    else:
        common_module = TensorDictModule(
            embedding_net(),
            in_keys=[("agents", "observation")],
            out_keys=[("hidden", "embedding"), ("hidden", "att_embedding")],
        )

    actor_module = TensorDictModule(
        actor_net(),
        in_keys=[
            ("hidden", "embedding"),
            ("hidden", "att_embedding"),
            ("agents", "observation", "valid_actions"),
        ],
        out_keys=[("agents", "logits")],
    )

    policy = ProbabilisticActor(
        module=actor_module,
        in_keys=("agents", "logits"),
        out_keys=[("agents", "action")],
        distribution_class=torch.distributions.categorical.Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        cache_dist=True,
        default_interaction_type=InteractionType.RANDOM,  # we sample actions randomly
    )

    critic_module = ValueOperator(
        critic_net(),
        in_keys=[("hidden", "embedding"), ("hidden", "att_embedding")],
        out_keys=["state_value"],
    )

    model = ActorValueOperator(common_module, policy, critic_module).to(device)

    if args.pretrained_network_path is not None:
        model_path = args.pretrained_network_path
        assert model_path.endswith("tar"), "Network format not known."
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loaded pretrained model from .tar file")

    # Set-Up loss module

    loss_module = ClipPPOLoss(
        actor=ProbabilisticTensorDictSequential(common_module, policy),
        critic=critic_module,
        critic_coef=args.vf_coef,
        clip_epsilon=args.clip_coef,
        entropy_coef=args.ent_coef,
        normalize_advantage=args.norm_adv,
        loss_critic_type=args.value_loss,
    )

    loss_module.set_keys(
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        advantage=("agents", "advantage"),
    )

    optim = torch.optim.Adam(
        loss_module.parameters(), lr=args.learning_rate, weight_decay=1e-7
    )

    if args.zero_tree_attributes:
        print("freeze trees")
        for param in common_module.tree_lstm.parameters():
            param.data = torch.zeros_like(param.data)
            param.requires_grad = False  # freeze
    global_step = 0

    for param in common_module.tree_lstm.parameters():
        print(param.requires_grad)

    print("\n \n Number of Parameters:\n")
    print(
        f'{"attr embedding:":<15} {sum(p.numel() for p in common_module.attr_embedding.parameters() if p.requires_grad):>10,}'
    )
    print(
        f'{"tree embedding:":<15} {sum(p.numel() for p in common_module.tree_lstm.parameters() if p.requires_grad):>10,}'
    )
    print(
        f'{"transformer:":<15} {sum(p.numel() for p in common_module.transformer.parameters() if p.requires_grad):>10,}'
    )
    print(
        f'{"policy net:":<15} {sum(p.numel() for p in model.get_policy_head().parameters() if p.requires_grad):>10,}'
    )
    print(
        f'{"value net:":<15} {sum(p.numel() for p in model.get_value_head().parameters() if p.requires_grad):>10,}'
    )
    print(
        f'{"total:":<15} {sum(p.numel() for p in model.parameters() if p.requires_grad):>10,}'
    )

    print("\n \n")
    # load learning curriculum
    if args.curriculum_path is not None:
        curriculums = open(args.curriculum_path)
        curriculums = json.load(curriculums)
        print(curriculums)
    else:
        curriculums = [
            {
                "map_height": args.map_height,
                "map_width": args.map_width,
                "num_agents": args.num_agents,
                "reward_coefs": None,
                "total_timesteps": args.total_timesteps,
                "num_steps": args.num_steps,
            },
        ]

    start_time = time.time()

    # Start PPO loop
    for curriculum in curriculums:
        print(f"Current curriculum settings: {curriculum}")
        args.batch_size = int(
            args.num_envs * curriculum["num_steps"]
        )  # size of data used for training
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        print(f"minibatch size: {args.minibatch_size}")
        print(f"batch size: {args.batch_size}")

        if args.map_name is None:

            def make_env() -> TorchRLRailEnv:
                """Generate a random rail environment"""
                td_env = TDRailEnv(
                    number_of_agents=curriculum["num_agents"],
                    width=curriculum["map_width"],
                    height=curriculum["map_height"],
                    rail_generator=SparseRailGen(
                        max_num_cities=3,
                        grid_mode=False,
                        max_rails_between_cities=2,
                        max_rail_pairs_in_city=2,
                    ),
                    line_generator=SparseLineGen(speed_ratio_map={1.0: 1}),
                    malfunction_generator=ParamMalfunctionGen(
                        MalfunctionParameters(
                            malfunction_rate=1 / 4500, min_duration=20, max_duration=50
                        )
                    ),
                    obs_builder_object=TreeCutils(31, 500),
                )

                td_env.set_reward_coef(curriculum["reward_coefs"])
                td_env.reset()
                return TorchRLRailEnv(td_env)

        else:

            def make_env() -> TorchRLRailEnv:
                """Return a TorchRL environment."""
                td_env = TDRailEnv(
                    number_of_agents=curriculum["num_agents"],
                    width=curriculum["map_width"],
                    height=curriculum["map_height"],
                    rail_generator=rail_from_grid_transition_map(
                        *generate_custom_rail(map_name=args.map_name)
                    ),
                    line_generator=SparseLineGen(
                        speed_ratio_map={1.0: 1},  # 0.5: '', 0.33: 1 / 4, 0.25: 1 / 4},
                        seed=1,
                    ),
                    malfunction_generator=ParamMalfunctionGen(
                        MalfunctionParameters(
                            malfunction_rate=1 / 4500, min_duration=20, max_duration=50
                        ),
                    ),
                    obs_builder_object=TreeCutils(31, 500),
                )
                td_env.set_reward_coef(curriculum["reward_coefs"])
                td_env.reset()
                return TorchRLRailEnv(td_env)

        if args.do_multisync:
            print("doing multisync")

            def make_serial_env():
                return ParallelEnv(args.num_envs, make_env)

            collector = MultiaSyncDataCollector(
                [make_serial_env, make_serial_env, make_serial_env],
                model,
                device=["cpu", "cpu", "cpu"],  # env runs on cpu
                storing_device=[device, device, device],
                frames_per_batch=args.batch_size,
                total_frames=curriculum["total_timesteps"],
                update_at_each_batch=True,
                max_frames_per_traj=-1,
                init_random_frames=args.batch_size,
            )
        else:
            env = ParallelEnv(args.num_envs, make_env)

            collector = SyncDataCollector(
                env,
                model,
                device="cpu",  # env runs on cpu
                storing_device=device,
                frames_per_batch=args.batch_size,
                total_frames=curriculum["total_timesteps"],
            )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(args.batch_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=args.minibatch_size,
        )

        start_rollout = time.time()

        collector.update_policy_weights_()

        for i_batch, tensordict_data in enumerate(collector):  # start training loops
            if args.do_multisync and i_batch < 3:
                print("warm up iteration")
                continue
            rollout_duration = time.time() - start_rollout
            training_start = time.time()

            global_step += args.batch_size

            assert tensordict_data[("next", "agents", "reward")].shape == torch.Size(
                [args.num_envs, curriculum["num_steps"], curriculum["num_agents"]]
            )
            assert tensordict_data[
                ("next", "agents", "observation", "agents_attr")
            ].shape == torch.Size(
                [args.num_envs, curriculum["num_steps"], curriculum["num_agents"], 83]
            )
            assert tensordict_data[("agents", "action")].shape == torch.Size(
                [args.num_envs, curriculum["num_steps"], curriculum["num_agents"]]
            )
            assert tensordict_data["state_value"].shape == torch.Size(
                [args.num_envs, curriculum["num_steps"]]
            )

            # track info of rollout
            if args.exp_name is not None:
                writer.add_scalar(
                    "rewards/min",
                    tensordict_data[("next", "agents", "reward")].min(),
                    global_step,
                )
                writer.add_scalar(
                    "rewards/mean",
                    tensordict_data[("next", "agents", "reward")].mean(),
                    global_step,
                )
                writer.add_scalar(
                    "rewards/max",
                    tensordict_data[("next", "agents", "reward")].max(),
                    global_step,
                )
                softmax_value = torch.sigmoid(
                    tensordict_data[("agents", "logits")]
                ).mean((0, 1, 2))
                writer.add_scalar(
                    "action_probs/do_nothing", softmax_value[0], global_step
                )
                writer.add_scalar("action_probs/left", softmax_value[1], global_step)
                writer.add_scalar("action_probs/forward", softmax_value[2], global_step)
                writer.add_scalar("action_probs/right", softmax_value[3], global_step)
                writer.add_scalar(
                    "action_probs/stop_moving", softmax_value[4], global_step
                )
                writer.add_scalar(
                    "charts/rollout_steps_per_second",
                    args.batch_size / rollout_duration,
                    global_step,
                )
                final_steps = tensordict_data[("next", "done")].squeeze()
                writer.add_scalar(
                    "charts/total_rollout_duration", rollout_duration, global_step
                )
                final_stats = tensordict_data[
                    ("next", "agents", "observation", "agents_attr")
                ][final_steps][:, :, (6, 41)].mean((0, 1))
                writer.add_scalar(
                    "stats/arrival_ratio",
                    final_stats[0],
                    global_step,
                )
                print(f"arrival ratio: {final_stats[0]}")
                print(f"deadlock ratio: {final_stats[1]}")
                writer.add_scalar(
                    "stats/deadlock_ratio",
                    final_stats[1],
                    global_step,
                )
                writer.add_scalar(
                    "action_frequency/do_nothing",
                    (tensordict_data[("agents", "action")] == 0).sum()
                    / (args.batch_size * curriculum["num_agents"]),
                    global_step,
                )
                writer.add_scalar(
                    "action_frequency/left",
                    (tensordict_data[("agents", "action")] == 1).sum()
                    / (args.batch_size * curriculum["num_agents"]),
                    global_step,
                )
                writer.add_scalar(
                    "action_frequency/forward",
                    (tensordict_data[("agents", "action")] == 2).sum()
                    / (args.batch_size * curriculum["num_agents"]),
                    global_step,
                )
                writer.add_scalar(
                    "action_frequency/right",
                    (tensordict_data[("agents", "action")] == 3).sum()
                    / (args.batch_size * curriculum["num_agents"]),
                    global_step,
                )
                writer.add_scalar(
                    "action_frequency/stop_moving",
                    (tensordict_data[("agents", "action")] == 4).sum()
                    / (args.batch_size * curriculum["num_agents"]),
                    global_step,
                )

            # Prepare rollout data for training
            tensordict_data = tensordict_data.to(device)
            tensordict_data.set(
                ("next", "agents", "reward"),
                tensordict_data.get(("next", "agents", "reward")).mean(-1),
            )  # we average rewards over all agents in an env step

            # Calculation of generalized advantage estimator
            # based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
            with torch.no_grad():
                lastgaelam = torch.zeros(args.num_envs).to(device)
                tensordict_data["advantage"] = torch.zeros_like(
                    tensordict_data["state_value"]
                ).to(
                    device
                )  # only one advantage per environment and step

                # get value of final observations
                next_val = model.get_value_operator()(
                    tensordict_data[("next")][:, curriculum["num_steps"] - 1]
                )

                for t in reversed(range(curriculum["num_steps"])):
                    if (
                        t == curriculum["num_steps"] - 1
                    ):  # for the last one, we get the special values for the last observation
                        nextnonterminal = ~tensordict_data[("next", "done")][
                            :, t
                        ].squeeze()
                        # print(f'shape nextonterminal: {nextnonterminal.shape}')
                        nextvalues = next_val["state_value"].squeeze()
                    else:
                        # for all other rollouts, we get the dones from the current t,
                        # and the values from the next t
                        nextnonterminal = ~tensordict_data[("next", "done")][
                            :, t
                        ].squeeze()
                        nextvalues = tensordict_data[("state_value")][
                            :, t + 1
                        ].squeeze()
                    delta = (
                        tensordict_data[("next", "agents", "reward")][:, t]
                        + args.gamma * nextvalues * nextnonterminal
                        - tensordict_data[("state_value")][:, t]
                    )

                    tensordict_data["advantage"][:, t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    ).flatten()

                tensordict_data["value_target"] = (
                    tensordict_data["advantage"] + tensordict_data["state_value"]
                )

            # now repeat the same advantage for each agent such that loss calculation works
            tensordict_data[("agents", "advantage")] = (
                tensordict_data["advantage"]
                .repeat_interleave(curriculum["num_agents"])
                .reshape(tensordict_data[("agents", "action")].shape)
                .unsqueeze(-1)
            )

            assert tensordict_data[("agents", "advantage")].shape == torch.Size(
                [args.num_envs, curriculum["num_steps"], curriculum["num_agents"], 1]
            )
            assert tensordict_data[("agents", "sample_log_prob")].shape == torch.Size(
                [args.num_envs, curriculum["num_steps"], curriculum["num_agents"]]
            )

            # Combine data from all parallel envs and send to buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            if args.exp_name is not None:
                approx_kl_minibatch = torch.zeros(args.num_minibatches)
                old_approx_kl_minibatch = torch.zeros(args.num_minibatches)
                clip_frac = torch.zeros(args.num_minibatches)

            for n_epoch in range(args.update_epochs):
                # print(f"epoch nr: {n_epoch}")
                for n_minibatch in range(args.num_minibatches):
                    subdata = replay_buffer.sample().clone()
                    # print(f'advantages: {subdata[("agents", "advantage")]}')

                    loss_vals = loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    if args.exp_name is not None:
                        # estimate KL divergence
                        # again based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
                        if n_epoch == args.update_epochs - 1:
                            original_data_logprobs = subdata[
                                ("agents", "sample_log_prob")
                            ].clone()
                            original_actions = subdata[("agents", "action")].clone()
                            with torch.no_grad():
                                updated_logits = model(subdata.clone())[
                                    ("agents", "logits")
                                ]
                                dist = torch.distributions.Categorical(
                                    logits=updated_logits
                                )
                                updated_data_logprobs = dist.log_prob(original_actions)
                            logratio = updated_data_logprobs - original_data_logprobs
                            approx_kl_minibatch[n_minibatch] = (
                                (logratio.exp() - 1) - logratio
                            ).mean()
                            old_approx_kl_minibatch[n_minibatch] = (-logratio).mean()
                            clip_frac[n_minibatch] = (
                                ((logratio.exp() - 1.0).abs() > args.clip_coef)
                                .float()
                                .mean()
                            )

                    loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), args.max_grad_norm
                    )

                    optim.step()
                    optim.zero_grad()

            training_duration = time.time() - training_start

            collector.update_policy_weights_()

            if args.exp_name is not None:
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value = loss_value / args.num_minibatches
                writer.add_scalar(
                    "charts/learning_rate", optim.param_groups[0]["lr"], global_step
                )
                writer.add_scalar(
                    "losses/value_loss", loss_vals["loss_critic"], global_step
                )
                writer.add_scalar(
                    "losses/policy_loss", loss_vals["loss_objective"], global_step
                )
                writer.add_scalar(
                    "losses/entropy", loss_vals["loss_entropy"], global_step
                )

                writer.add_scalar("losses/total_loss", loss_value, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar(
                    "charts/training_speed",
                    args.batch_size * args.update_epochs / training_duration,
                    global_step,
                )
                writer.add_scalar(
                    "charts/total_training_duration", training_duration, global_step
                )

                writer.add_scalar(
                    "losses/approx_kl_minibatch_mean",
                    approx_kl_minibatch.mean(),
                    global_step,
                )
                writer.add_scalar(
                    "losses/old_approx_kl_approch_kl_minibatch_mean",
                    old_approx_kl_minibatch.mean(),
                    global_step,
                )
                writer.add_scalar("losses/clipfrac_mean", clip_frac.mean(), global_step)
                writer.add_scalar(
                    "losses/approx_kl_minibatch_max",
                    approx_kl_minibatch.max(),
                    global_step,
                )
                writer.add_scalar(
                    "losses/old_approx_kl_approch_kl_minibatch_max",
                    old_approx_kl_minibatch.max(),
                    global_step,
                )
                writer.add_scalar("losses/clipfrac_max", clip_frac.max(), global_step)

                if (global_step / args.batch_size) % 10 == 0:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                        },
                        f"model_checkpoints/{run_name}/{run_name}_"
                        + str(global_step)
                        + ".tar",
                    )

            start_rollout = time.time()
