import msgParser

class CarControl(object):
    '''
    An object holding all the control parameters of the car
    '''
    # TODO range check on set parameters

    def __init__(self, accel=0.0, brake=0.0, gear=1, steer=0.0, clutch=0.0, focus=0, meta=0):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        
        self.actions = None
        
        self.accel = accel
        self.brake = brake
        self.gear = gear
        self.steer = steer
        self.clutch = clutch
        self.focus = focus
        self.meta = meta
    
    def toMsg(self):
        self.actions = {}
        
        self.actions['accel'] = [self.accel]
        self.actions['brake'] = [self.brake]
        self.actions['gear'] = [self.gear]
        self.actions['steer'] = [self.steer]
        self.actions['clutch'] = [self.clutch]
        self.actions['focus'] = [self.focus]
        self.actions['meta'] = [self.meta]
        
        return self.parser.stringify(self.actions)
    
    def to_dict(self):
        '''Return all control parameters as a dictionary for logging'''
        result = {
            'accel': self.accel if self.accel is not None else 0.0,
            'brake': self.brake if self.brake is not None else 0.0,
            'gear': self.gear if self.gear is not None else 0,
            'steer': self.steer if self.steer is not None else 0.0,
            'clutch': self.clutch if self.clutch is not None else 0.0,
            'focus': self.focus if self.focus is not None else 0,
            'meta': self.meta if self.meta is not None else 0
        }
        print(f"CarControl.to_dict gear: {result['gear']}")  # Debug print
        return result
    
    def setAccel(self, accel):
        self.accel = accel
    
    def getAccel(self):
        return self.accel
    
    def setBrake(self, brake):
        self.brake = brake
    
    def getBrake(self):
        return self.brake
    
    def setGear(self, gear):
        self.gear = gear
    
    def getGear(self):
        return self.gear
    
    def setSteer(self, steer):
        self.steer = steer
    
    def getSteer(self):
        return self.steer
    
    def setClutch(self, clutch):
        self.clutch = clutch
    
    def getClutch(self):
        return self.clutch
    
    def setMeta(self, meta):
        self.meta = meta
    
    def getMeta(self):
        return self.meta