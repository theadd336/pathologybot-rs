#!/usr/local/bin/python
import logging
import sys

from multiprocessing import Queue, Process

from pathologybot_py.model import ImpalaModel, ModelSize, ModelMode
from pathologybot_py.gym.pathology import PathologyGym


def setup_logging():
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def run_learner(incoming_queue: Queue, outgoing_queues: list[Queue]):
    learner = ImpalaModel(ModelSize.Smol, mode=ModelMode.Learner, name="Learner")
    learner.train(None, incoming_queue, outgoing_queues)
    learner.save_weights_to("./learned_weights")


def run_actor(
    trajectory_queue: Queue,
    weights_queues: list[Queue],
    actor_id: int,
):
    actor = ImpalaModel(ModelSize.Smol, mode=ModelMode.Actor, name=f"Actor{actor_id}")
    gym = PathologyGym(test_mode=True)
    actor.train(gym, trajectory_queue, weights_queues, actor_id, epochs=50)


def main():
    setup_logging()
    num_actors = 7
    weights_queues = [Queue() for _ in range(num_actors)]
    trajectory_queue = Queue()
    learner_p = Process(target=run_learner, args=(trajectory_queue, weights_queues))
    learner_p.start()
    actor_p_list = [
        Process(target=run_actor, args=(trajectory_queue, weights_queues, i))
        for i in range(num_actors)
    ]
    for actor in actor_p_list:
        actor.start()
    learner_p.join()
    for actor in actor_p_list:
        actor.join()

    player = ImpalaModel(ModelSize.Smol, name="player")
    player.load_weights_from("./learned_weights")
    gym = PathologyGym(test_mode=True)
    for _ in range(30):
        state = gym.reset()
        player.evaluate(gym, state)


if __name__ == "__main__":
    main()
