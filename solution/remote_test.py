import argparse
import os
import traceback
from time import sleep

from eval_env import TestEnvWrapper
from plfActor import Actor
from utils import VideoWriter, debug_show

parser = argparse.ArgumentParser()
parser.add_argument("--render", "-r", action="store_true", default=False)
parser.add_argument("--save-videos", "-s", action="store_true")
parser.add_argument(
    "--fps", type=float, default=30, help="frames per second (default 10)"
)
parser.add_argument("--tests-folder", default="./debug-environments/")
parser.add_argument("--policy-folder", default="./policy/")

args = parser.parse_args()


os.environ["AICROWD_TESTS_FOLDER"] = args.tests_folder

actor_50 = Actor(os.sep.join([args.policy_folder, "phase-III-50.pt"]))
actor_80 = Actor(os.sep.join([args.policy_folder, "phase-III-80.pt"]))
actor_100 = Actor(os.sep.join([args.policy_folder, "phase-III-100.pt"]))
actor_200 = Actor(os.sep.join([args.policy_folder, "phase-III-200.pt"]))


def get_actor(n_agents):
    if n_agents <= 50:
        return actor_50
    elif n_agents <= 80:
        return actor_80
    elif n_agents <= 100:
        return actor_100
    else:
        return actor_200


railenv = TestEnvWrapper()
episode = 0

while True:
    print("==============")
    episode += 1
    print("[INFO] EPISODE_START : {}".format(episode))
    # NO WAY TO CHECK service/self.evaluation_done in client

    obs = railenv.reset()
    actor = get_actor(railenv.env.number_of_agents)
    if obs is False:
        # The remote env returns False as the first obs
        # when it is done evaluating all the individual episodes
        print("[INFO] DONE ALL, BREAKING")
        break

    # get test_no and level
    test_file = railenv.remote_client.current_env_path
    test_no = test_file.split("/")[-2]
    level = test_file.split("/")[-1].split(".")[0]

    # create video writer
    if args.save_videos:
        dir = os.path.join("replay_video", test_no)
        os.makedirs(dir, exist_ok=True)
        videofile = os.path.join(dir, f"{level}.mp4")
        video_writer = VideoWriter(videofile, args.fps)

    while True:
        va = railenv.get_valid_actions()
        n_agents = railenv.remote_client.env.number_of_agents
        action = actor.get_actions(obs, va, n_agents)
        try:
            obs, all_rewards, done = railenv.step(action)

            if args.save_videos:
                frame = debug_show(railenv.remote_client.env, mode="rgb_array")
                video_writer.write(frame)
            if args.render:
                debug_show(railenv.remote_client.env)
                sleep(0.1)
        except:
            traceback.print_exc()
            # print("[ERR] DONE BUT step() CALLED")

        # break
        if done["__all__"]:
            if args.save_videos:
                video_writer.close()
                print(f"[INFO] Save replay to {videofile}")

            arrival_ratio, total_reward, norm_reward = railenv.final_metric()
            print(f"[INFO] EPISODE_DONE : {episode}")
            print(f"[INFO] TOTAL_REW: {total_reward}")
            print(f"[INFO] NORM_REW: {norm_reward:.4f}")
            print(f"[INFO] ARR_RATIO: {arrival_ratio*100:.2f}%")
            break

print("Evaluation Complete...")
print(railenv.submit())
