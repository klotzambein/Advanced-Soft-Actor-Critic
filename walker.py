import opensim
import math

def list_elements(model):
    muscleSet = model.getMuscles()
    forceSet = model.getForceSet()
    bodySet = model.getBodySet()
    jointSet = model.getJointSet()
    markerSet = model.getMarkerSet()
    contactGeometrySet = model.getContactGeometrySet()
    actuatorSet = model.getActuators()
    print("JOINTS")
    for i in range(jointSet.getSize()):
        print(i, jointSet.get(i).getName())
    print("\nBODIES")
    for i in range(bodySet.getSize()):
        print(i, bodySet.get(i).getName())
    print("\nMUSCLES")
    for i in range(muscleSet.getSize()):
        print(i, muscleSet.get(i).getName())
    print("\nFORCES")
    for i in range(forceSet.getSize()):
        print(i, forceSet.get(i).getName())
    print("\nMARKERS")
    for i in range(markerSet.getSize()):
        print(i, markerSet.get(i).getName())
    print("\nACTUATORS")
    for i in range(actuatorSet.getSize()):
        print(i, actuatorSet.get(i).getName())

def create_and_add_controller(model):
    ctrlr = opensim.PrescribedController()

    actuatorSet = model.getActuators()
    for j in range(actuatorSet.getSize()):
        actuator = actuatorSet.get(j)
        ctrlr.addActuator(actuator)
        print(opensim.CoordinateActuator.safeDownCast(actuator).get_coordinate())

        func = opensim.Constant(0.0)
        ctrlr.prescribeControlForActuator(j, func)
        
    model.addController(ctrlr)
    return ctrlr

def set_action(controller, idx, value):
    function_set = controller.get_ControlFunctions()

    func = opensim.Constant.safeDownCast(function_set.get(idx))
    func.setValue(float(value))

model = opensim.Model("envs/opensim/TransfProst-VR-NoMuscles.osim")
# model = opensim.Model("envs/opensim/WalkerModel.osim")

# list_elements(model)

model.setUseVisualizer(True)

controller = create_and_add_controller(model)



state = model.initSystem()
state = model.initializeState()

model.equilibrateMuscles(state)
state.setTime(0)
manager = opensim.Manager(model)
manager.setIntegratorAccuracy(0.001)
manager.initialize(state)

class QTripplet:
    def __init__(self, p, v, a):
        self.pos = p
        self.vel = v
        self.acc = a
    
    def __str__(self):
        return "(%f, %f, %f)" % (self.pos, self.vel, self.acc)

def stablePDWithVel(state, next_target, delta_time):
    kp = 1
    kd = 1
    control_p = -kp * (state.pos + delta_time * state.vel - next_target.pos)
    control_d = -kd * (state.vel + delta_time * state.acc - next_target.vel)
    return control_p + control_d

def stablePD(state, next_target, delta_time):
    kp = 0.9
    kd = 0.0001
    control_p = -kp * (state.pos + delta_time * state.vel - next_target)
    control_d = -kd * (state.vel + delta_time * state.acc)
    return control_p + control_d

delta_time = 1.0 / 200.0

for i in range(1, 10000):
    target = math.sin(i * math.pi * delta_time) - 1.0

    actuatorSet = model.getActuators()
    for j in range(actuatorSet.getSize()):
        actuator = opensim.CoordinateActuator.safeDownCast(actuatorSet.get(j))
        coord = actuator.getCoordinate()

        pos = coord.getValue(state)
        vel = coord.getSpeedValue(state)
        acc = coord.getAccelerationValue(state)
        coord = QTripplet(pos, vel, acc)

        ctrl = stablePD(coord, target if j in [2, 3] else 0, delta_time)
        # print(str(coords[0]) + " -> " + str(ctrl))
        set_action(controller, j, ctrl)
    
    state = manager.integrate(i * delta_time)

