#!/usr/local/bin/python
import logging
import numpy as np
import sys

from multiprocessing import Queue, Process

from pathologybot_py.model import ImpalaModel, ModelSize, ModelMode
from pathologybot_py.gym.pathology import PathologyGym


def setup_logging():
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)


def run_learner(incoming_queue: Queue, outgoing_queues: list[Queue]):
    learner = ImpalaModel(ModelSize.Smol, mode=ModelMode.Learner)
    learner.train(None, incoming_queue, outgoing_queues)


def run_actor(
    trajectory_queue: Queue,
    weights_queues: list[Queue],
    actor_id: int,
):
    actor = ImpalaModel(ModelSize.Smol, mode=ModelMode.Actor)
    gym = PathologyGym(test_mode=True)
    actor.train(gym, trajectory_queue, weights_queues, actor_id)


def main():
    setup_logging()
    num_actors = 1
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


if __name__ == "__main__":
    main()
