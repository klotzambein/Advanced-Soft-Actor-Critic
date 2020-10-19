import logging
import os
import threading
import time
from collections import deque
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import numpy as np

import algorithm.config_helper as config_helper

from . import constants as C
from .proto import evolver_pb2, evolver_pb2_grpc, learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .utils import PeerSet, rpc_error_inspector


def update_nn_variables(stub, mean, std):
    nn_variables = [np.random.normal(mean[i], std[i]) for i in range(len(mean))]
    stub.update_nn_variables(nn_variables)


class Evolver:
    def __init__(self, root_dir, config_dir, args):
        self.logger = logging.getLogger('ds.evolver')

        (self.config,
         self.net_config,
         model_abs_dir,
         config_abs_dir) = self._init_config(root_dir, config_dir, args)

        self._learners = dict()
        self._learner_lock = threading.Lock()

        self._last_update_nn_variable = time.time()
        self._saved_nn_variables_mean = None
        self._saved_nn_variables_std = None

        self._update_nn_variables_executors = ThreadPoolExecutor(10)

        try:
            self._run()
        except KeyboardInterrupt:
            self.logger.warning('KeyboardInterrupt in _run')
            self.close()

    def _init_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        default_config_file_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_file_path,
                                                           config_abs_path,
                                                           args.config)

        if args.evolver_host is not None:
            config['net_config']['evolver_host'] = args.evolver_host
        if args.evolver_port is not None:
            config['net_config']['evolver_port'] = args.evolver_port
        if args.name is not None:
            config['base_config']['name'] = args.name

        config['base_config']['name'] = config_helper.generate_base_name(config['base_config']['name'], 'ds')
        model_abs_dir = Path(root_dir).joinpath('models',
                                                config['base_config']['scene'],
                                                config['base_config']['name'])
        os.makedirs(model_abs_dir)

        if args.logger_in_file:
            config_helper.set_logger(Path(model_abs_dir).joinpath('evolver.log'))

        config_helper.save_config(config, model_abs_dir, 'config_ds.yaml')

        config_helper.display_config(config, self.logger)

        return (config['base_config'],
                config['net_config'],
                model_abs_dir,
                config_abs_dir)

    def _learner_connected(self, peer, connected):
        with self._learner_lock:
            if connected:
                self._learners[peer] = {
                    'rewards': deque(maxlen=self.config['evolver_cem_length']),
                    'selected': 0
                }
            else:
                del self._learners[peer]

    def _post_reward(self, reward, peer):
        with self._learner_lock:
            self._learners[peer]['rewards'].append(reward)

            # If there is only one learner, return
            if len(self._learners) <= 1:
                return

            rewards_length_map = map(lambda x: len(x['rewards']), self._learners.values())
            # All learners have evaluated more than evolver_cem_length times
            # or all learners have evaluated more than evolver_cem_min_length times and
            #   it has been more than evolver_cem_time mins since last evolution
            if all([l == self.config['evolver_cem_length'] for l in rewards_length_map]) or \
                    (all([l >= self.config['evolver_cem_min_length'] for l in rewards_length_map]) and
                     time.time() - self._last_update_nn_variable >= self.config['evolver_cem_time'] * 60):

                # Sort learners by the mean of evaluated rewards
                learner_reward = [(k, float(np.mean(v['rewards']))) for k, v in self._learners.items()]
                learner_reward.sort(key=lambda i: i[1], reverse=True)

                # Select top evolver_cem_best learners and get their nn variables
                best_size = int(len(learner_reward) * self.config['evolver_cem_best'])
                best_size = max(best_size, 1)

                best_learners = [i[0] for i in learner_reward[:best_size]]
                nn_variables_list = list()
                for learner in best_learners:
                    self._learners[learner]['selected'] += 1
                    stub = self.servicer.get_learner_stub(learner)
                    if stub:
                        nn_variables_list.append(stub.get_nn_variables())

                # Calculate the mean and std of best_size variables of learners
                mean = [np.mean(i, axis=0) for i in zip(*nn_variables_list)]
                std = [np.minimum(np.std(i, axis=0), 1.) for i in zip(*nn_variables_list)]

                self._saved_nn_variables_mean, self._saved_nn_variables_std = mean, std

                # Dispatch all nn variables
                for learner in self._learners.keys():
                    stub = self.servicer.get_learner_stub(learner)
                    if stub:
                        self._update_nn_variables_executors.submit(update_nn_variables,
                                                                   stub, mean, std)

                    self._learners[learner]['rewards'].clear()

                self._last_update_nn_variable = time.time()

                _best_learner_ids = [str(self.servicer.get_learner_id(l)) for l in best_learners]
                self.logger.info(f'Selected {",".join(_best_learner_ids)} learners')

                _learner_id_selecteds = [(str(self.servicer.get_learner_id(l)), v['selected']) for l, v in self._learners.items()]
                _learner_id_selecteds.sort(key=lambda x: x[1], reverse=True)
                _learner_id_selecteds = [f'{i[0]}({i[1]})' for i in _learner_id_selecteds]
                self.logger.info(f'Learner id (selected): {", ".join(_learner_id_selecteds)}')

                std = [(np.min(s), np.mean(s), np.max(s)) for s in std]
                _min, _mean, _max = [np.mean(s) for s in zip(*std)]
                self.logger.info(f'Variables std: {_min:.2f}, {_mean:.2f}, {_max:.2f}')

    def _get_nn_variables(self):
        if self._saved_nn_variables_mean is None:
            return None

        mean, std = self._saved_nn_variables_mean, self._saved_nn_variables_std
        return [np.random.normal(mean[i], std[i]) for i in range(len(mean))]

    def _run(self):
        self.servicer = EvolverService(self.config['name'],
                                       self.config['max_actors_each_learner'],
                                       self._learner_connected,
                                       self._post_reward,
                                       self._get_nn_variables)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=C.MAX_THREAD_WORKERS))
        evolver_pb2_grpc.add_EvolverServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f'[::]:{self.net_config["evolver_port"]}')
        self.server.start()
        self.logger.info(f'Evolver server is running on [{self.net_config["evolver_port"]}]...')

        self.server.wait_for_termination()

    def close(self):
        self.server.stop(0)

        self.logger.warning('Closed')


class LearnerStubController:
    def __init__(self, host, port):
        self._channel = grpc.insecure_channel(f'{host}:{port}', [
            ('grpc.max_reconnect_backoff_ms', C.MAX_RECONNECT_BACKOFF_MS),
            ('grpc.max_send_message_length', C.MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', C.MAX_MESSAGE_LENGTH)
        ])
        self._stub = learner_pb2_grpc.LearnerServiceStub(self._channel)

        self._logger = logging.getLogger('ds.evolver.learner_stub')

    @rpc_error_inspector
    def get_nn_variables(self):
        response = self._stub.GetNNVariables(Empty())
        return [proto_to_ndarray(v) for v in response.variables]

    @rpc_error_inspector
    def update_nn_variables(self, variables):
        self._stub.UpdateNNVariables(learner_pb2.NNVariables(
            variables=[ndarray_to_proto(v) for v in variables]))

    def close(self):
        self._channel.close()


class EvolverService(evolver_pb2_grpc.EvolverServiceServicer):
    def __init__(self, name, max_actors_each_learner, learner_connected,
                 post_reward,
                 get_nn_variables):
        self.name = name
        self.max_actors_each_learner = max_actors_each_learner

        self._learner_connected = learner_connected
        self._post_reward = post_reward
        self._get_nn_variables = get_nn_variables

        self._logger = logging.getLogger('ds.evolver.service')
        self._peer_set = PeerSet(self._logger)

        self._learner_lock = threading.Lock()
        self._learner_id = 0
        self._learner_actors = dict()
        self._actor_learner = dict()

    def _record_peer(self, context):
        peer = context.peer()

        def _unregister_peer():
            with self._learner_lock:
                if peer in self._learner_actors:
                    info = self._peer_set.get_info(peer)
                    _id = info['id']
                    del self._learner_actors[peer]
                    self._learner_connected(peer, connected=False)
                    self._logger.warning(f'Learner {peer} (id={_id}) disconnected')
                elif peer in self._actor_learner:
                    learner_peer = self._actor_learner[peer]
                    if learner_peer in self._learner_actors:
                        self._learner_actors[learner_peer].remove(peer)
                    del self._actor_learner[peer]
                    self._logger.warning(f'Actor {peer} disconnected')
            self._peer_set.disconnect(peer)

        context.add_callback(_unregister_peer)
        self._peer_set.connect(peer)

    @property
    def learners(self):
        with self._learner_lock:
            return list(self._learner_actors.keys())

    def get_learner_stub(self, peer):
        info = self._peer_set.get_info(peer)
        if info:
            return info['stub']

    def get_learner_id(self, peer):
        info = self._peer_set.get_info(peer)
        if info:
            return info['id']

    def display_learner_actors(self):
        with self._learner_lock:
            info = 'Learner-actors:'
            for learner in self._learner_actors:
                learner_info = self._peer_set.get_info(learner)
                learner_id = learner_info['id']
                learner_host = learner_info['learner_host']
                learner_port = learner_info['learner_port']
                info += f'\n{learner} ({learner_host}:{learner_port} id={learner_id}): {len(self._learner_actors[learner])}'

        self._logger.info(info)

    # From learner and actor
    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def RegisterLearner(self, request: evolver_pb2.RegisterLearnerRequest, context):
        peer = context.peer()
        learner_id = self._learner_id

        self._peer_set.add_info(peer, {
            'id': learner_id,
            'learner_host': request.learner_host,
            'learner_port': request.learner_port,
            'replay_host': request.replay_host,
            'replay_port': request.replay_port,
            'stub': LearnerStubController(request.learner_host, request.learner_port)
        })
        with self._learner_lock:
            self._learner_actors[peer] = set()
        self._learner_connected(peer, connected=True)

        self._logger.info(f'Learner {peer} (id={learner_id}) registered')

        self.display_learner_actors()

        self._learner_id += 1

        return evolver_pb2.RegisterLearnerResponse(name=self.name, id=learner_id)

    def RegisterActor(self, request, context):
        peer = context.peer()

        with self._learner_lock:
            if len(self._learner_actors) == 0:
                self._logger.info(f'Actor {peer} register failed, no learner exists')
                return evolver_pb2.RegisterActorResponse(succeeded=False)

            assigned_learner = sorted(self._learner_actors.items(),
                                      key=lambda t: len(t[1]))[0][0]

            if len(self._learner_actors[assigned_learner]) == self.max_actors_each_learner:
                self._logger.info(f'Actor {peer} register failed, all learners have max actors')
                return evolver_pb2.RegisterActorResponse(succeeded=False)

            self._learner_actors[assigned_learner].add(peer)
            self._actor_learner[peer] = assigned_learner

        info = self._peer_set.get_info(assigned_learner)

        learner_id = info['id']
        learner_host, learner_port = info['learner_host'], info['learner_port']
        replay_host, replay_port = info['replay_host'], info['replay_port']

        log = f'Actor {peer} registered to ' +\
            f'learner (id={learner_id}) {learner_host}:{learner_port}, ' +\
            f'replay {replay_host}:{replay_port}'
        self._logger.info(log)

        self.display_learner_actors()

        return evolver_pb2.RegisterActorResponse(succeeded=True,
                                                 learner_host=learner_host,
                                                 learner_port=learner_port,
                                                 replay_host=replay_host,
                                                 replay_port=replay_port)

    # From learner
    def PostReward(self, request: evolver_pb2.PostRewardToEvolverRequest, context):
        self._post_reward(float(request.reward), context.peer())

        return Empty()

    # From learner
    def GetNNVariables(self, request, context):
        variables = self._get_nn_variables()
        if variables is None:
            return evolver_pb2.GetNNVariablesResponse(succeeded=False)
        else:
            return evolver_pb2.GetNNVariablesResponse(
                succeeded=True,
                variables=[ndarray_to_proto(v) for v in variables])
