import numpy as np

class ExtendedKalmanFilter:
    """
    A class to implement the Extended Kalman Filter (EKF).

    Attributes:
        stateDim (int): Dimension of the state vector.
        measDim (int): Dimension of the measurement vector.
        Q (ndarray): Process noise covariance matrix.
        R (ndarray): Measurement noise covariance matrix.
        x (ndarray): State estimate vector.
        P (ndarray): State covariance matrix.
    """
    
    def __init__(self, stateDim, measDim, processNoiseCov, measNoiseCov, bubbleGrowth, dt = 1):
        """
        Initializes the EKF with specified dimensions and noise covariances.

        Args:
            stateDim (int): Dimension of the state vector.
            measDim (int): Dimension of the measurement vector.
            processNoiseCov (ndarray): Process noise covariance matrix.
            measNoiseCov (ndarray): Measurement noise covariance matrix.
            dt (float): Time step for state transition
            bubbleGrowth (bool): Toggle bubble growth module
        """
        self.dt = dt
        self.stateDim = stateDim
        self.measDim = measDim
        self.Q = processNoiseCov
        self.R = measNoiseCov

        # initial states
        self.x = np.zeros((stateDim, 1))  # initial state
        self.P = np.eye(stateDim)  # initial state covariance

        # implement bubble growth
        self.bubbleGrowth = bubbleGrowth

        # initialize confidence for each tracker
        self.confidence = 0.5

    def predict(self, u):
        """
        Predicts the next state and updates the state covariance.

        Args:
            u (ndarray): Control input vector.
        """
        # predict the state
        self.x = self.stateTransFunc(self.x, u)
        
        # Jacobian of state transition function
        F = self.jacobianStateFunc()

        # predict covariance
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q

    def update(self, z):
        """
        Updates the state estimate based on the measurement.

        Args:
            z (ndarray): Measurement vector.
        """
        # Jacobian measurement function
        H = self.jacobianMeasFunc()

        # innovation residual
        y = z - self.measFunc(self.x)

        # innovation covariance
        S = np.dot(H, np.dot(self.P, H.T)) + self.R

        # reguralize matrix
        S += np.eye(self.measDim) * 1e-6

        # near-optimal Kalman gain
        K = np.dot(self.P, np.dot(H.T, np.linalg.pinv(S)))

        # update state and covariance estimate
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.stateDim) - np.dot(K, H)), self.P)


    def stateTransFunc(self, x, u):
        """
        Defines the state transition function.

        Args:
            x (ndarray): Current state vector.
            u (ndarray): Control input vector.

        Returns:
            ndarray: Predicted state vector.
        """
        posX, posY, velX, velY = x[:4].flatten()

        if self.bubbleGrowth:
            size = x[4] if self.stateDim > 4 else 0
            growthRate = u[0] if u.size > 0 else 0
            return np.array([
                [posX + velX * self.dt],  # Updated position X
                [posY + velY * self.dt],  # Updated position Y
                [velX],                   # Velocity X remains unchanged
                [velY],                   # Velocity Y remains unchanged
                [size + growthRate * self.dt]  # Updated bubble size
            ])
        
        else:
            return np.array([
                [posX + velX * self.dt],
                [posY + velY * self.dt],
                [velX],
                [velY]
            ])             

    def measFunc(self, x):
        """
        Defines the measurement function.

        Args:
            x (ndarray): Current state vector.

        Returns:
            ndarray: Measured state vector.
        """
        if self.bubbleGrowth:
            return x[:3]  # position and size
        
        return x[:2] # position

    def jacobianStateFunc(self):
        """
        Computes the Jacobian of the state transition function

        Args:
            None

        Returns:
            ndarray: Jacobian matrix of the state transition function
        """
        if self.bubbleGrowth:
            return np.array([
            [1, 0, self.dt, 0, 0],  # df/dx
            [0, 1, 0, self.dt, 0],  # df/dy
            [0, 0, 1, 0, 0],        # df/dvx
            [0, 0, 0, 1, 0],        # df/dvy
            [0, 0, 0, 0, 1]         # df/d(size) 
        ]) 

        return np.array([
            [1, 0, self.dt, 0],  # df/dx
            [0, 1, 0, self.dt],  # df/dy
            [0, 0, 1, 0],        # df/dvx
            [0, 0, 0, 1]         # df/dvy
        ])
    
    def jacobianMeasFunc(self):
        """
        Computes the Jacobian of the measurement function

        Args:
            None

        Returns:
            ndarray: Jacobian matrix of the measurement function
        """
        if self.bubbleGrowth:
            return np.array([
                [1, 0, 0, 0, 0], # dh/dx
                [0, 1, 0, 0, 0],  # dh/dy
                [0, 0, 0, 0, 1]  # dh/d(self)
            ])

        return np.array([
            [1, 0, 0, 0], # dh/dx
            [0, 1, 0, 0]  # dh/dy
        ]) 
    