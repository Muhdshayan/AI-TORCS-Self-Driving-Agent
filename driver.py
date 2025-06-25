import numpy as np
import joblib
import tensorflow as tf
import msgParser
import carState
import carControl
import os

class Driver:
    def __init__(self, stage):
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.stage = stage

        model_dir = './model'
        self.tflite_model_path = os.path.join(model_dir, 'torcs_model.tflite')
        self.input_scaler_path = os.path.join(model_dir, 'input_scaler.pkl')
        self.output_scaler_path = os.path.join(model_dir, 'output_scaler.pkl')
        self.means_path = os.path.join(model_dir, 'means.npy')
        self.stds_path = os.path.join(model_dir, 'stds.npy')
        self.feature_file = os.path.join(model_dir, 'input_features.txt')

        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.output_idx = self.interpreter.get_output_details()[0]['index']

        self.scaler = joblib.load(self.input_scaler_path)
        self.y_scaler = joblib.load(self.output_scaler_path)
        self.means = np.load(self.means_path)
        self.stds = np.load(self.stds_path)
        with open(self.feature_file) as f:
            self.feature_names = [l.strip() for l in f if l.strip()]

        # PD controller params
        self.prev_error = 0.0
        self.kp = 1.0
        self.kd = 0.2

        # Gear shifting parameters
        self.shift_up_rpms = [2500, 3000, 3500, 4000, 4500]
        self.shift_down_rpms = [1500, 2000, 2500, 3000, 3500]
        self.speed_thresholds = [50, 100, 150, 200, 250]
        self.downshift_speed = [40, 90, 140, 190, 240]
        self.ticks_per_shift = 10
        self.tick_in_gear = 0

        # Accel smoothing
        self.prev_accel = 0.0
        self.accel_ramp = 0.05
        self.stuck_ticks = 0

        # Recovery thresholds
        self.forward_clear = 20.0  # distance ahead
        self.rear_clear = 20.0     # distance behind (approx side sensors)

    def init(self):
        angles = [-90 + 10 * i for i in range(19)]
        return self.parser.stringify({'init': angles})

    def drive(self, sensor_msg):
        self.state.setFromMsg(sensor_msg)
        S = self.state

        # Build features
        feat = {'Angle': S.angle, 'Damage': S.damage, 'FuelLevel': S.fuel,
                'Gear_in': S.gear, 'RacePosition': S.racePos, 'RPM': S.rpm,
                'SpeedX': S.speedX, 'SpeedY': S.speedY, 'SpeedZ': S.speedZ,
                'TrackPosition': S.trackPos, 'Z': S.z}
        for i, v in enumerate(S.opponents): feat[f'Opponent_{i+1}'] = v
        for i, v in enumerate(S.track): feat[f'Track_{i+1}'] = v
        for i, v in enumerate(S.wheelSpinVel): feat[f'WheelSpinVelocity_{i+1}'] = v

        # Normalize
        x = np.array([feat[n] for n in self.feature_names], dtype=np.float32)
        x = (x - self.means) / self.stds
        x = self.scaler.transform(x.reshape(1, -1)).astype(np.float32)

        # Predict
        self.interpreter.set_tensor(self.input_idx, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_idx)[0]
        accel_pred, brake_pred, clutch_pred, _, steer_pred = \
            self.y_scaler.inverse_transform(y.reshape(1, -1))[0]

        # Sensor readings
        forward = np.mean([feat['Track_9'], feat['Track_10'], feat['Track_11']])
        rear = np.mean([feat['Track_1'], feat['Track_19']])
        off_track = abs(S.trackPos) > 1.0
        straight = abs(S.angle) < 0.01 and abs(S.trackPos) < 0.05 and forward > 90

        # Stuck logic
        if S.speedX < 1 and forward < 3:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        # Recovery when stuck
        if self.stuck_ticks > 10:
            # Choose direction based on clear path
            if off_track:
                if rear > self.rear_clear and (rear > forward):
                    # Reverse if rear is clearer
                    gear = -1; accel, brake, clutch = 0.5, 0.0, 0.0
                elif forward > self.forward_clear:
                    # Forward if front is clearer
                    gear = 1; accel, brake, clutch = 1.0, 0.0, 0.0
                else:
                    # Default to reverse
                    gear = -1; accel, brake, clutch = 0.5, 0.0, 0.0
            else:
                # On track but stuck, reverse first
                if rear > self.rear_clear:
                    gear = -1; accel, brake, clutch = 0.5, 0.0, 0.0
                else:
                    gear = 1; accel, brake, clutch = 1.0, 0.0, 0.0
            steer = 0.0
        else:
            # Regular driving
            if straight:
                target = 1.0 if S.speedX < 350 else 0.9
                delta = target - self.prev_accel
                accel = self.prev_accel + np.sign(delta) * self.accel_ramp
                accel = np.clip(accel, 0, target)
                brake, clutch = 0.0, 0.0
            else:
                accel = float(np.clip(accel_pred, 0, 1))
                brake = float(np.clip(brake_pred, 0, 1))
                clutch = float(np.clip(clutch_pred, 0, 1))
            self.prev_accel = accel

            # Gear shifting
            gear = S.gear
            self.tick_in_gear += 1
            if self.tick_in_gear > self.ticks_per_shift:
                if gear < 6 and (S.rpm > self.shift_up_rpms[gear-1] or S.speedX > self.speed_thresholds[gear-1]):
                    gear += 1; self.tick_in_gear = 0
                elif gear > 1 and (S.rpm < self.shift_down_rpms[gear-2] and S.speedX < self.downshift_speed[gear-2]):
                    gear -= 1; self.tick_in_gear = 0

            # Steering PD
            error = S.trackPos
            d_err = error - self.prev_error
            pd = -(self.kp*error + self.kd*d_err)
            self.prev_error = error
            if straight:
                steer = float(np.clip(pd, -1, 1))
            else:
                steer = 0.3*pd + 0.7*np.clip(steer_pred, -1, 1)
                steer = float(np.clip(steer, -1, 1))

        # Apply controls
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        self.control.setClutch(clutch)
        self.control.setGear(gear)
        self.control.setSteer(steer)
        return self.control.toMsg()

    def onShutdown(self): pass
    def onRestart(self): pass
