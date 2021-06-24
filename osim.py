import math
import numpy as np
import os
import gym
import opensim
import random
from opensim import ScalarActuator
from opensim import CoordinateActuator


# OpenSim interface
# The main purpose of this class is to provide wrap all
# the necessery elements of OpenSim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class
class OsimModel(object):
    # Initialize simulation
    stepsize = 0.002

    model_path = None
    state = None
    brain = None

    state_desc_istep = None
    state_desc = None
    integrator_accuracy = None

    visualize = False

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy=5e-5, stepsize=0.01):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()
        self.actuatorSet = self.model.getActuators()

        # self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle and actuator
        for j in range(self.actuatorSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.actuatorSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

        for j in range(self.muscleSet.getSize()):
            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        # Try to add constant functions for the motors as well

        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i, self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i, self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i, self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i, self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i, self.markerSet.get(i).getName())
        print("\nACTUATORS")
        for i in range(self.actuatorSet.getSize()):
            print(i, self.actuatorSet.get(i).getName())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError(
                "NaN passed in the activation vector. Values in [0,1] interval are required.")

        #action = np.clip(np.array(action), 0.0, 1.0)
        self.last_action = action

        brain = opensim.PrescribedController.safeDownCast(
            self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(float(action[j]))

    """
    Directly modifies activations in the current state.
    """

    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError(
                "NaN passed in the activation vector. Values in [0,1] interval are required.")

        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])

        self.reset_manager()

    """
    Get activations in the given state.
    """

    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def get_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        res['pose'] = Pose()
        res['pose'].set_from_q_vector(self.state.getQ())

        # Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in
                                      range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in
                                      range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in
                                      range(joint.numCoordinates())]

        # Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[
                i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(
                self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(
                self.state).get(1).get(i) for i in range(3)]

            res["body_pos_rot"][name] = [
                body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(
                self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(
                self.state).get(0).get(i) for i in range(3)]

        # Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        # Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(
                self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(
                self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(
                self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(
                self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html

        # Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[
                i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[
                i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[
                i] for i in range(3)]

        # Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[
            i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[
            i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [
            self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(
                self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.actuatorSet.getSize()

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)

        self.reset_manager()

    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        t = self.state.getTime()
        t += self.stepsize

        # Integrate till the new endtime
        self.state = self.manager.integrate(t)

    def step(self, action):
        self.actuate(action)
        self.integrate()

    def initializeState(self):
        self.state = self.model.initializeState()


# RUG TFP Model
# This environment provides basic interface to the transfemoral amputee model developed at RUG
class RUGTFPEnv(OsimModel):

    # 11.7769 + 9.30139 + 3.7075 + 0.1 + 1.25 + 0.21659 + 4.5 + 0.8199 + 0.77 + 0.710828809 + 34.2366 = 67.389708809
    MASS = 67.38971  # Add up all the body segement mass.
    G = 9.80665  # from gait1dof22muscle

    dict_muscle = {'abd_r': 'HAB_R',
                   'add_r': 'HAD_R',
                   'iliopsoas_r': 'HFL_R',
                   'glut_max_r': 'GLU_R',
                   'hamstrings_r': 'HAM_R',
                   'rect_fem_r': 'RF_r',
                   'vasti_r': 'VAS_R',
                   'bifemsh_r': 'BFSH_R',
                   'gastroc_r': 'GAS_R',
                   'soleus_r': 'SOL_R',
                   'tib_ant_r': 'TA_R',
                   'abd_l': 'HAB_L',
                   'add_l': 'ADD_L',
                   'glut_max_l': 'GLU_L',
                   'iliopsoas_l': 'HFL_L'}

    right_leg_MUS = ['HAB_R', 'HAD_R', 'HFL_R', 'GLU_R', 'HAM_R',
                     'RF_R', 'VAS_R', 'BFSH_R', 'GAS_R', 'SOL_R', 'TA_R']  # 11 muscles
    right_leg_mus = ['abd_r', 'add_r', 'iliopsoas_r', 'glut_max_r', 'hamstrings_r',
                     'rect_fem_r', 'vasti_r', 'bifemsh_r', 'gastroc_r', 'soleus_r', 'tib_ant_r']

    left_leg_MUS = ['HAB_L', 'ADD_L', 'GLU_L', 'HFL_L']  # 4 muscles
    left_leg_mus = ['abd_l', 'add_l', 'glut_max_l', 'iliopsoas_l']

    # muscle order in action
    # HAB_R, HAD_R, HFL_R, GLU_R, HAM_R, RF_R, VAS_R, BFSH_R, GAS_R, SOL_R, TA_R, HAB_L, ADD_L, GLU_L, HFL_L, KNE_ACT, ANK_ACT

    def __init__(self, model_name="", visualize=True, integrator_accuracy=5e-5, stepsize=0.01):
        self.model_path = f"/home/robin/Desktop/rug-opensim-rl/osim/models/{model_name}"
        super().__init__(visualize=visualize, model_path=self.model_path,
                         integrator_accuracy=integrator_accuracy, stepsize=stepsize)

        self.Fmax = {}
        self.lopt = {}
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            self.Fmax[leg] = {}
            self.lopt[leg] = {}
            for MUS, mus in zip(['HAB_R', 'HAD_R', 'HFL_R', 'GLU_R', 'HAM_R', 'RF_R', 'VAS_R', 'BFSH_R', 'GAS_R', 'SOL_R', 'TA_R', 'HAB_L', 'ADD_L', 'GLU_L', 'HFL_L'],
                                ['abd_r', 'add_r', 'iliopsoas_r', 'glut_max_r', 'hamstrings_r', 'rect_fem_r', 'vasti_r', 'bifemsh_r', 'gastroc_r', 'soleus_r', 'tib_ant_r', 'abd_l', 'add_l', 'glut_max_l', 'iliopsoas_l']):
                try:
                    muscle = self.muscleSet.get('{}'.format(mus))
                    Fmax = muscle.getMaxIsometricForce()
                    lopt = muscle.getOptimalFiberLength()

                    self.Fmax[leg][MUS] = muscle.getMaxIsometricForce()
                    self.lopt[leg][MUS] = muscle.getOptimalFiberLength()
                except Exception as e:
                    # print(e) # Harmless exception to catch the unused muscles
                    pass

        # Actuator Optimal Force
        # Manual way of getting the optimal force from the knee and ankle actuators
        actuator_names = ['knee_actuator', 'ankle_actuator']
        self.Fmax['l_leg']['KNE_ACT'] = CoordinateActuator.safeDownCast(
            self.actuatorSet.get('{}'.format(actuator_names[0]))).getOptimalForce()
        self.Fmax['l_leg']['ANK_ACT'] = CoordinateActuator.safeDownCast(
            self.actuatorSet.get('{}'.format(actuator_names[1]))).getOptimalForce()

    def reset(self, init_pose: 'Pose'):
        self.state = self.model.initializeState()

        qt = init_pose.getQ()
        ut = init_pose.getU()

        state = self.get_state()
        q = state.getQ()
        u = state.getQDot()

        for i in range(17):
            q[i] = qt[i]
            u[i] = ut[i]

        state.setQ(q)
        state.setU(u)
        self.set_state(state)

        self.model.equilibrateMuscles(self.state)

        self.state.setTime(0)

        self.reset_manager()

    def step(self, action):
        super().step(action)

    def get_observation_space_size(self):
        return 88

    def get_state_desc(self):
        d = super().get_state_desc()
        self.model.realizeAcceleration(self.state)

        # Actuators
        try:
            d["actuators"] = {}
            d["actuators"]["knee"] = {}
            knee_scalar = ScalarActuator.safeDownCast(
                self.actuatorSet.get('knee_actuator'))
            d["actuators"]["knee"]["speed"] = knee_scalar.getSpeed(self.state)
            d["actuators"]["knee"]["control"] = knee_scalar.getControl(
                self.state)
            d["actuators"]["knee"]["actuation"] = knee_scalar.getActuation(
                self.state)
            d["actuators"]["knee"]["power"] = knee_scalar.getPower(self.state)
            d["actuators"]["knee"]["stress"] = knee_scalar.getStress(
                self.state)

            d["actuators"]["ankle"] = {}
            ankle_scalar = ScalarActuator.safeDownCast(
                self.actuatorSet.get('ankle_actuator'))
            d["actuators"]["ankle"]["speed"] = ankle_scalar.getSpeed(
                self.state)
            d["actuators"]["ankle"]["control"] = ankle_scalar.getControl(
                self.state)
            d["actuators"]["ankle"]["actuation"] = ankle_scalar.getActuation(
                self.state)
            d["actuators"]["ankle"]["power"] = ankle_scalar.getPower(
                self.state)
            d["actuators"]["ankle"]["stress"] = ankle_scalar.getStress(
                self.state)
        except Exception as e:
            print(e)

        return d

    def get_observation_dict(self, state_desc):
        obs_dict = {}

        # pelvis state (in local frame)
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['pos'] = state_desc['body_pos']['pelvis']

        obs_dict['pelvis']['pitch'] = state_desc['joint_pos']['ground_pelvis'][0]
        obs_dict['pelvis']['roll'] = state_desc['joint_pos']['ground_pelvis'][1]
        obs_dict['pelvis']['yaw'] = state_desc['joint_pos']['ground_pelvis'][2]

        yaw = state_desc['joint_pos']['ground_pelvis'][2]

        dx_local, dy_local = rotate_frame(state_desc['body_vel']['pelvis'][0],
                                          state_desc['body_vel']['pelvis'][2],
                                          yaw)

        dz_local = state_desc['body_vel']['pelvis'][1]
        obs_dict['pelvis']['vel'] = [dx_local,  # (+) forward
                                     -dy_local,  # (+) leftward
                                     dz_local,  # (+) upward
                                     # (+) pitch angular velocity
                                     -state_desc['joint_vel']['ground_pelvis'][0],
                                     # (+) roll angular velocity
                                     state_desc['joint_vel']['ground_pelvis'][1],
                                     state_desc['joint_vel']['ground_pelvis'][2]]  # (+) yaw angular velocity

        return obs_dict

    def get_leg_observations(self, side: str, state_desc):
        yaw = state_desc['joint_pos']['ground_pelvis'][2]
        leg_obs = {}

        grf = [f / (self.MASS * self.G) for f in
               state_desc['forces']['foot_{}'.format(side)][0:3]]  # forces normalized by bodyweight
        # grm = [m / (self.MASS * self.G) for m in
        #         state_desc['forces']['foot_{}'.format(side)][3:6]]  # forces normalized by bodyweight
        grfx_local, grfy_local = rotate_frame(-grf[0], -grf[2], yaw)
        if side == 'l':
            grfy_local *= -1
        leg_obs['ground_reaction_forces'] = [grfx_local,  # (+) forward
                                             # (+) lateral (outward)
                                             grfy_local,
                                             -grf[1]]  # (+) upward

        # joint angles
        leg_obs['joint'] = {}
        # (+) hip abduction
        leg_obs['joint']['hip_abd'] = - \
            state_desc['joint_pos'][f'hip_{side}'][1]
        leg_obs['joint']['hip'] = - \
            state_desc['joint_pos'][f'hip_{side}'][0]  # (+) extension
        # (+) extension
        leg_obs['joint']['knee'] = state_desc['joint_pos'][f'knee_{side}'][0]
        leg_obs['joint']['ankle'] = - \
            state_desc['joint_pos'][f'ankle_{side}'][0]  # (+) extension
        # joint angular velocities
        leg_obs['d_joint'] = {}
        leg_obs['d_joint']['hip_abd'] = -state_desc['joint_vel'][f'hip_{side}'][
            1]  # (+) hip abduction
        leg_obs['d_joint']['hip'] = - \
            state_desc['joint_vel'][f'hip_{side}'][0]  # (+) extension
        # (+) extension
        leg_obs['d_joint']['knee'] = state_desc['joint_vel'][f'knee_{side}'][0]
        leg_obs['d_joint']['ankle'] = - \
            state_desc['joint_vel'][f'ankle_{side}'][0]  # (+) extension

        # muscles
        if side == 'r':
            MUS_list = self.right_leg_MUS
            mus_list = self.right_leg_mus
            leg = 'r_leg'
        else:
            MUS_list = self.left_leg_MUS
            mus_list = self.left_leg_mus
            leg = 'l_leg'
        for MUS, mus in zip(MUS_list,
                            mus_list):
            leg_obs[MUS] = {}
            leg_obs[MUS]['f'] = state_desc['muscles'][mus]['fiber_force'] / \
                self.Fmax[leg][MUS]
            leg_obs[MUS]['l'] = state_desc['muscles'][mus]['fiber_length'] / \
                self.lopt[leg][MUS]
            leg_obs[MUS]['v'] = state_desc['muscles'][mus]['fiber_velocity'] / \
                self.lopt[leg][MUS]

        # actuators
        if side == 'l':
            leg_obs['force'] = {}
            leg_obs['actuator'] = {}
            leg_obs['actuator']['knee'] = {}
            leg_obs['actuator']['ankle'] = {}

            leg_obs['force']['knee'] = state_desc["actuators"]["knee"]["control"] * \
                self.Fmax['l_leg']['KNE_ACT']  # get instantaneous force
            leg_obs['actuator']['knee']['speed'] = state_desc["actuators"]["knee"]["speed"]
            leg_obs['actuator']['knee']['control'] = state_desc['actuators']['knee']['control']
            leg_obs['actuator']['knee']['power'] = state_desc['actuators']['knee']['power']
            leg_obs['actuator']['knee']['stress'] = state_desc['actuators']['knee']['stress']
            leg_obs['actuator']['knee']['actuation'] = state_desc['actuators']['knee']['actuation']

            leg_obs['force']['ankle'] = state_desc["actuators"]["ankle"]["control"] * \
                self.Fmax['l_leg']['ANK_ACT']  # get instataneous force
            leg_obs['actuator']['ankle']['speed'] = state_desc['actuators']['ankle']['speed']
            leg_obs['actuator']['ankle']['control'] = state_desc['actuators']['ankle']['control']
            leg_obs['actuator']['ankle']['power'] = state_desc['actuators']['ankle']['power']
            leg_obs['actuator']['ankle']['stress'] = state_desc['actuators']['ankle']['stress']
            leg_obs['actuator']['ankle']['actuation'] = state_desc['actuators']['ankle']['actuation']

        return leg_obs


def rotate_frame(x, y, theta):
    x_rot = np.cos(theta) * x - np.sin(theta) * y
    y_rot = np.sin(theta) * x + np.cos(theta) * y
    return x_rot, y_rot


def rotate_frame_3D(x, y, z, axis, theta):
    if axis == 'x':
        coord_axis = opensim.CoordinateAxis(0)
    elif axis == 'y':
        coord_axis = opensim.CoordinateAxis(1)
    elif axis == 'z':
        coord_axis = opensim.CoordinateAxis(2)
    else:
        raise Exception("Coordinate axis should be either x,y or z")

    # Rotation matrix
    rot_matrix = opensim.Rotation(np.deg2rad(theta), coord_axis)
    v = opensim.Vec3(x, y, z)
    rotated_frame = rot_matrix.multiply(v)
    x_rot = rotated_frame[0]
    y_rot = rotated_frame[1]
    z_rot = rotated_frame[2]

    return x_rot, y_rot, z_rot


class Pose:
    """
    This is the Layout of the Q vector:
    00: pelvis tilt 
    01: pelvis list
    02: pelvis rotation
    03: pelvis x
    04: pelvis y
    05: pelvis z
    06: hip flexion right 
    07: hip abduction right
    08: (hip ... right)
    09: hip flexion left
    10: hip abduction left
    11: (hip ... left)
    12: (lumbar ext)
    13: knee right 
    14: ankle right
    15: knee left
    16: ankle left
    """

    def __init__(self):
        self.pose = np.zeros((17, 2))
        # The lumbar extension is always -5 degree
        self.pose[16, 0] = -5.0 * np.pi / 180

    def set_pelvis_euler_angles(self, tilt: float, list: float, rotation: float):
        """
        This sets the rotation of the pelvis relative to the ground. All angles are in radians.
        :param tilt: pitch (positive moves the head back and the feet forward)
        :param list: roll (positive moves the head to the right and the feet to the left)
        :param rotation: yaw (positive turns the model to the left)
        """
        self.pose[0:3, 0] = [tilt, list, rotation]

    def set_pelvis_position(self, x: float, y: float, z: float):
        self.pose[3:6, 0] = [x, y, z]

    def set_leg_left(self, hip_flexion: float, hip_abduction: float, knee: float, ankle: float):
        self.pose[9, 0] = hip_flexion
        self.pose[10, 0] = hip_abduction
        self.pose[15, 0] = knee
        self.pose[16, 0] = ankle

    def set_leg_right(self, hip_flexion: float, hip_abduction: float, knee: float, ankle: float):
        self.pose[6, 0] = hip_flexion
        self.pose[7, 0] = hip_abduction
        self.pose[13, 0] = knee
        self.pose[14, 0] = ankle

    def set_from_dict(self, pose, rotation_scale: float = 1.0):
        rs = rotation_scale
        self.set_pelvis_position(
            pose['pelvis_tx'], pose['pelvis_ty'], pose['pelvis_tz'])
        self.set_pelvis_euler_angles(
            pose['pelvis_tilt'] * rs, pose['pelvis_list'] * rs, pose['pelvis_rotation'] * rs)
        self.set_leg_left(pose['hip_flexion_l'] * rs, pose['hip_adduction_l']
                          * rs, pose['knee_angle_l'] * rs, pose['ankle_angle_l'] * rs)
        self.set_leg_right(pose['hip_flexion_r'] * rs, pose['hip_adduction_r']
                           * rs, pose['knee_angle_r'] * rs, pose['ankle_angle_r'] * rs)

    def set_from_dict_degrees(self, pose):
        self.set_from_dict(pose, rotation_scale=np.pi / 180)

    def set_from_q_vector(self, q):
        for i in range(17):
            self.pose[i, 0] = q[i]

    def compute_velocities(self, next: 'Pose', delta_time: float):
        """
        Given the next pose and the time between the current pose and the next
        pose this function will compute the appropriate velocities to go from
        the current pose to the next one.
        """
        self.pose[:, 1] = (next.pose[:, 0] - self.pose[:, 0]) / delta_time

    def is_in_bounds(self, bounds_center: 'Pose', max_angle: float, max_distance: float):
        delta = abs(self.pose[:, 0] - bounds_center.pose[:, 0])

        pelvis_rot = np.all(delta[0:3] < max_angle)
        pelvis_pos = np.all(delta[3:6] < max_distance)
        hip_right = np.all(delta[6:8] < max_angle)
        hip_left = np.all(delta[9:11] < max_angle)
        rest = np.all(delta[13:17] < max_angle)

        # print(pelvis_rot, pelvis_pos, hip_right, hip_left, rest, delta[13:17] / max_angle)

        return pelvis_rot and pelvis_pos and hip_right and hip_left and rest

    def getQ(self):
        return self.pose[:, 0]

    def getU(self):
        return self.pose[:, 1]

    def toString(self):
        return ",".join([str(x) for x in self.getQ().tolist()])
