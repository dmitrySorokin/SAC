import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union
import time

import openvr

from robel.components.tracking.virtual_reality.device import VrDevice
from robel.components.tracking.virtual_reality.poses import VrPoseBatch

from model import GaussianPolicy

ROTATE = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
         ]
    )


class FCModel(nn.Module):
    def __init__(self, space_shape, hidden_size=64):
        super().__init__()

        self.lin1 = nn.Linear(space_shape, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, space_shape)

        self.nonlin = nn.LeakyReLU(0.2)

    def forward(self, batch):
        batch = self.nonlin(self.lin1(batch))
        batch = self.nonlin(self.lin2(batch))
        batch = self.lin3(batch)

        return batch

def policy_factory(model_path, env):
    agent = GaussianPolicy(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        256,
        env.action_space
    ).to('cpu')

    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def policy(obs):
        state = torch.FloatTensor(obs).unsqueeze(0)
        _, _, action = agent.sample(state)
        return action.detach().cpu().numpy()[0]

    return policy


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class VrClient:
    """Communicates with a VR device."""

    def __init__(self):
        self._vr_system = None
        self._devices = []
        self._device_serial_lookup = {}
        self._device_index_lookup = {}
        self._last_pose_batch = None
        self._plot = None

        # Attempt to start OpenVR.
        if not openvr.isRuntimeInstalled():
            raise OSError('OpenVR runtime not installed.')

        self._vr_system = openvr.init(openvr.VRApplication_Other)

    def close(self):
        """Cleans up any resources used by the client."""
        if self._vr_system is not None:
            openvr.shutdown()
            self._vr_system = None

    def get_device(self, identifier: Union[int, str]) -> VrDevice:
        """Returns the device with the given name."""
        identifier = str(identifier)
        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        if identifier in self._device_serial_lookup:
            return self._device_serial_lookup[identifier]

        self.discover_devices()
        if (identifier not in self._device_index_lookup
                and identifier not in self._device_serial_lookup):
            raise ValueError(
                'Could not find device with name or index: {} (Available: {})'
                .format(identifier, sorted(self._device_serial_lookup.keys())))

        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        return self._device_serial_lookup[identifier]

    def discover_devices(self) -> List[VrDevice]:
        """Returns and caches all connected devices."""
        self._device_index_lookup.clear()
        self._device_serial_lookup.clear()
        devices = []
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device = VrDevice(self._vr_system, device_index)
            if not device.is_connected():
                continue
            devices.append(device)
            self._device_index_lookup[str(device.index)] = device
            self._device_serial_lookup[device.get_serial()] = device
        self._devices = devices
        return devices

    def get_poses(self, time_from_now: float = 0.0,
                  update_plot: bool = True) -> VrPoseBatch:
        """Returns a batch of poses that can be queried per device.

        Args:
            time_from_now: The seconds into the future to read poses.
            update_plot: If True, updates an existing plot.
        """
        pose_batch = VrPoseBatch(self._vr_system, time_from_now)
        self._last_pose_batch = pose_batch
        if update_plot and self._plot and self._plot.is_open:
            self._plot.refresh()
        return pose_batch

    def __enter__(self):
        """Enables use as a context manager."""
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.close()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.close()


def print_observation(observation):
    print('='*100)
    print('pos:', observation[:3])
    print('euler: ', observation[3:6])
    print('joint_pos', observation[6:18])
    print('vel', observation[18:21])
    print('angular_vel: ', observation[21:24])
    print('joint_vel: ', observation[24:36])
    print('last_action: ', observation[36:48])
