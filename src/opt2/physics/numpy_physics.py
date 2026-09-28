import numpy as np

class NumpyPhysics:
    def __init__(self, params):
        self.p = params
        self.g = params.get('g', np.array([0, 0, -9.81]))
        
        # Cache static arrays to prevent reallocation overhead
        self._zeros3 = np.zeros(3)
        self._z_axis = np.array([0, 0, 1])
        self._eye3 = np.eye(3)
        
        self._r = self.p['ball_r']

    def _flight_dynamics(self, x):
        vel, ang_vel = x[6:9], x[9:12]
        v_norm = np.linalg.norm(vel)
        acc = self.g - self.p['C_D'] * v_norm * vel + self.p['C_L'] * np.cross(ang_vel, vel)
        return np.concatenate((vel, ang_vel, acc, self._zeros3))

    def rk4_step(self, x, dt):
        k1 = dt * self._flight_dynamics(x)
        k2 = dt * self._flight_dynamics(x + k1/2)
        k3 = dt * self._flight_dynamics(x + k2/2)
        k4 = dt * self._flight_dynamics(x + k3)
        return x + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    def check_table_contact(self, x):
        return (np.abs(x[0]) <= self.p['table_l'] / 2 and 
                np.abs(x[1]) <= self.p['table_w'] / 2 and 
                x[2] <= self._r)

    def table_reset(self, x_in, x_last):
        r, mu, ep = self._r, self.p['mu'], self.p['epsilon_t']
        vt = np.array([x_in[6] - r * x_in[10], x_in[7] + r * x_in[9], 0])
        alpha = mu * (1 + ep) * np.abs(x_in[8]) / (np.linalg.norm(vt) + 1e-8)
        
        if alpha < 0.4:
            A_v = np.diag([1 - alpha, 1 - alpha, -ep])
            B_v = np.array([[0, alpha*r, 0], [-alpha*r, 0, 0], [0, 0, 0]])
            A_w = np.array([[0, -3*alpha/(2*r), 0], [3*alpha/(2*r), 0, 0], [0, 0, 0]])
            B_w = np.diag([1 - 3*alpha/2, 1 - 3*alpha/2, 1])
        else:
            A_v = np.diag([0.6, 0.6, -ep])
            B_v = np.array([[0, 0.4*r, 0], [-0.4*r, 0, 0], [0, 0, 0]])
            A_w = np.array([[0, -0.6/r, 0], [0.6/r, 0, 0], [0, 0, 0]])
            B_w = np.diag([0.4, 0.4, 1])

        v_new = A_v @ x_in[6:9] + B_v @ x_in[9:12]
        w_new = A_w @ x_in[6:9] + B_w @ x_in[9:12]
        return np.concatenate((x_last[0:6], v_new, w_new))

    def _euler_to_rot(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])

    def check_racket_contact(self, ball_x, racket_x):
        R = self._euler_to_rot(*racket_x[3:6])
        diff = ball_x[0:3] - racket_x[0:3]
        normal = R @ self._z_axis
        return np.dot(diff, normal) <= 0 and np.linalg.norm(diff) <= self.p['racket_r']

    def racket_reset(self, ball_x, racket_x, ball_last):
        r, k_pv, k_pw, ep = self._r, self.p['k_pv'], self.p['k_pw'], self.p['epsilon_r']
        R = self._euler_to_rot(*racket_x[3:6])
        
        A_v = np.diag([1 - k_pv, 1 - k_pv, -ep])
        B_v = np.array([[0, k_pv*r, 0], [-k_pv*r, 0, 0], [0, 0, 0]])
        A_v_bar = R @ A_v @ R.T

        A_w = np.array([[0, -k_pw*r, 0], [k_pw*r, 0, 0], [0, 0, 0]])
        B_w = np.diag([1 - k_pw*(r**2), 1 - k_pw*(r**2), 1])
        A_w_bar = R @ A_w @ R.T

        v_out = (self._eye3 - A_v_bar) @ racket_x[6:9] + A_v_bar @ ball_x[6:9] + R @ B_v @ R.T @ ball_x[9:12]
        w_out = (self._eye3 - A_w_bar) @ racket_x[6:9] + A_w_bar @ ball_x[6:9] + R @ B_w @ R.T @ ball_x[9:12]
        return np.concatenate((ball_last[0:6], v_out, w_out))

    def step(self, state, racket1_x=None, racket2_x=None, dt=0.01):
        """Advances the simulation by 1 dt, resolving discrete collisions."""
        next_state = self.rk4_step(state, dt)
        
        if self.check_table_contact(next_state) and not self.check_table_contact(state):
            next_state = self.table_reset(next_state, state)
            
        elif racket1_x is not None and self.check_racket_contact(next_state, racket1_x) and not self.check_racket_contact(state, racket1_x):
            next_state = self.racket_reset(next_state, racket1_x, state)
            
        elif racket2_x is not None and self.check_racket_contact(next_state, racket2_x) and not self.check_racket_contact(state, racket2_x):
            next_state = self.racket_reset(next_state, racket2_x, state)
            
        return next_state