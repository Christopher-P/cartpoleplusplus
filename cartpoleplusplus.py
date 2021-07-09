"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/book/code/pole.c
"""
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p2
from pybullet_utils import bullet_client as bc
from pkg_resources import parse_version

logger = logging.getLogger(__name__)


class CartPoleBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False, discrete_actions=True):
        # start the bullet physics server
        self._renders = renders
        self._discrete_actions = discrete_actions
        self._render_height = 200
        self._render_width = 320
        self._physics_client_id = -1
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.4  # 2.4
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2, np.finfo(np.float32).max])

        # Environmental params
        self.force_mag = 15
        self.timeStep = 0.02

        # Object definitions
        self.cartpole = None
        self.ground = None
        self.blocks = None
        self.walls = None
        self.state = None

        if self._discrete_actions:
            self.action_space = spaces.Discrete(5)
        else:
            action_dim = 1
            action_high = np.array([self.force_mag] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        #    self.reset()
        self.viewer = None
        self._configure()

        return None

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p = self._p
        if self._discrete_actions:
            force = self.force_mag if action == 1 else -self.force_mag
        else:
            force = action[0]

        # based on action decide the x and y forces
        fx = fy = 0
        if action == 0:
            pass
        elif action == 1:
            fx = self.force_mag
        elif action == 2:
            fx = -self.force_mag
        elif action == 3:
            fy = self.force_mag
        elif action == 4:
            fy = -self.force_mag
        else:
            raise Exception("unknown discrete action [%s]" % action)
        p.applyExternalForce(self.cartpole, -1, (fx, fy, 0), (0, 0, 0), p.WORLD_FRAME)
        p.stepSimulation()

        done = False
        reward = 1.0
        # print("state=",self.state)
        return self.get_state(), reward, done, {}

    def reset(self):
        # Create client if it doesnt exist
        if self._physics_client_id < 0:
            self.generate_world()

        # Reset client params now that it is created
        p = self._p
        randstate = list(self.np_random.uniform(low=-0.05, high=0.05, size=(6,)))
        p.resetJointStateMultiDof(self.cartpole, 0, targetValue=randstate[2:5] + [1], targetVelocity=[0, 0, 0])
        return self.get_state()

    # Used to generate the initial world state
    def generate_world(self):
        # Create bullet physics client
        if self._renders:
            self._p = bc.BulletClient(connection_mode=p2.GUI)
        else:
            self._p = bc.BulletClient()
        self._physics_client_id = self._p._client

        # Load world simulation
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        # Load world objects
        self.ground = p.loadURDF("models/ground.urdf")
        self.cartpole = p.loadURDF("models/cartpole.urdf")
        self.walls = p.loadURDF("models/walls.urdf")

        nb_blocks = 4
        self.blocks = [None] * nb_blocks
        for i in range(nb_blocks):
            self.blocks[i] = p.loadURDF("models/block.urdf")

        # Set 0 friction on ground
        p.changeDynamics(self.ground, -1, restitution=0.0, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)

        # Set walls to be bouncy
        p.changeDynamics(self.walls, -1, restitution=1.0, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)
        p.changeDynamics(self.walls, 0, restitution=1.0, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)
        p.changeDynamics(self.walls, 1, restitution=1.0, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)
        p.changeDynamics(self.walls, 2, restitution=1.0, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)
        p.changeDynamics(self.walls, 3, restitution=1.0, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0)

        # Set blocks to be bouncy
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0)

        # This big line sets the spehrical joint on the pole to loose
        p.setJointMotorControlMultiDof(self.cartpole, 0, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=1,
                                       force=[0, 0, 0])

        # Set blocks to not overlap
        collide = False
        for i in self.blocks:
            for j in self.blocks:
                # Turn off block collisions
                p.setCollisionFilterPair(j, i, -1, -1, collide)

            # Turn of cart and pole collision with blocks
            p.setCollisionFilterPair(self.cartpole, i, -1, -1, collide)
            p.setCollisionFilterPair(self.cartpole, i, 0, -1, collide)

        # Set block posistions
        min_dist = 1
        cart_pos, _ = p.getBasePositionAndOrientation(self.cartpole)
        cart_pos = np.asarray(cart_pos)
        for i in self.blocks:
            pos = np.asarray(list(self.np_random.uniform(low=-4.5, high=4.5, size=(2,))) + [0.05])
            while np.linalg.norm(cart_pos - pos) < min_dist:
                pos = np.asarray(list(self.np_random.uniform(low=-9.0, high=9.0, size=(2,))) + [0.01])
            p.resetBasePositionAndOrientation(i, pos, [0,0,0,1])

        # Set block velocities
        for i in self.blocks:
            vel = np.asarray(list(self.np_random.uniform(low=-10.0, high=10.0, size=(2,))) + [0.0])
            p.resetBaseVelocity(i, vel, [0, 0, 0])
        return None

    # Unified function for getting state information
    def get_state(self):
        cart_json = self.get_object_json(self.cartpole)
        print(cart_json)
        pole_json = self.state_to_json(self.cartpole, 0)

        return {'cart': cart_json, 'pole': pole_json}

    def get_object_json(self, object_id):
        # Internal vars
        state = {}
        round_amount = 6

        # Handle pos, ori
        pos, ori = self._p.getBasePositionAndOrientation(object_id)

        state['x_position'] = round(pos[0], round_amount)
        state['y_position'] = round(pos[1], round_amount)
        state['z_position'] = round(pos[2], round_amount)

        state['x_quaternion'] = round(ori[0], round_amount)
        state['y_quaternion'] = round(ori[1], round_amount)
        state['z_quaternion'] = round(ori[2], round_amount)
        state['w_quaternion'] = round(ori[3], round_amount)

        # Handle velocity
        vel, ang = self._p.getBaseVelocity(object_id)
        state['x_velocity'] = round(vel[0], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        state['z_velocity'] = round(vel[2], round_amount)

        state['x_angular_velocity'] = round(ang[0], round_amount)
        state['y_angular_velocity'] = round(ang[1], round_amount)
        state['z_angular_velocity'] = round(ang[2], round_amount)
        state['w_angular_velocity'] = round(ang[2], round_amount)

        return state

    # TODO: Implement
    def get_joint_json(self):

        return None

    def render(self, mode='human', close=False):
        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        base_pos = [0, 0, 0]
        self._cam_dist = 2
        self._cam_pitch = 0.3
        self._cam_yaw = 0
        if (self._physics_client_id >= 0):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(self._render_width) /
                                                                    self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def configure(self, args):
        pass

    def eulerToQuaternion(self, yaw, pitch, roll):
        qx = np.sin(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) + np.cos(yaw / 2) * np.cos(pitch / 2) * np.sin(
            roll / 2)
        qy = np.sin(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) + np.cos(yaw / 2) * np.sin(pitch / 2) * np.sin(
            roll / 2)
        qz = np.cos(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) - np.sin(yaw / 2) * np.cos(pitch / 2) * np.sin(
            roll / 2)
        qw = np.cos(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) - np.sin(yaw / 2) * np.sin(pitch / 2) * np.sin(
            roll / 2)

        return (qx, qy, qz, qw)

    # Return state in json format
    def state_to_json(self, body_id, joint_id):

        p = self._p
        state = dict()
        round_amount = 6

        # Position and orientation, the other two not used
        pos, vel, jRF, aJMT = p.getJointStateMultiDof(body_id, joint_id)

        # Position
        if len(pos) == 4:
            state['x_quaternion'] = round(pos[0], round_amount)
            state['y_quaternion'] = round(pos[1], round_amount)
            state['z_quaternion'] = round(pos[2], round_amount)
            state['w_quaternion'] = round(pos[3], round_amount)
        else:
            state['x_position'] = round(pos[0], round_amount)
            state['y_position'] = round(pos[1], round_amount)
            if len(pos) == 3:
                state['z_position'] = round(pos[2], round_amount)
            else:
                state['z_position'] = 0.0

        # Velocity
        state['x_velocity'] = round(vel[0], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        if len(pos) == 3:
            state['z_velocity'] = round(vel[2], round_amount)
        else:
            state['z_velocity'] = 0.0

        return state

    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1


class CartPoleContinuousBulletEnv(CartPoleBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        # start the bullet physics server
        CartPoleBulletEnv.__init__(self, renders, discrete_actions=False)