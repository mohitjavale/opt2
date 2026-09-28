import numpy as np
import viser
from scipy.spatial.transform import Rotation

class ViserRenderer:
    def __init__(self, entities_cfg, render_cfg):
        self.ent_cfg = entities_cfg
        self.rnd_cfg = render_cfg
        
        # Launches the web server (default: http://localhost:8080)
        self.server = viser.ViserServer()
        self._setup_scene()

    def _setup_scene(self):
        t_c = self.ent_cfg.table.color
        n_c = self.ent_cfg.net.color
        b_c = self.ent_cfg.ball.color
        r1_c = self.ent_cfg.racket.color1
        r2_c = self.ent_cfg.racket.color2

        # Static objects (Viser uses full dimensions: l, w, h)
        self.server.scene.add_box(
            "/table",
            dimensions=(self.ent_cfg.table.l, self.ent_cfg.table.w, self.ent_cfg.table.h),
            color=t_c[:3],
            position=(0, 0, -self.ent_cfg.table.h / 2)
        )
        
        self.server.scene.add_box(
            "/net",
            dimensions=(self.ent_cfg.net.l, self.ent_cfg.net.w, self.ent_cfg.net.h),
            color=n_c[:3],
            position=(0, 0, self.ent_cfg.net.h / 2)
        )
        
        # Dynamic objects (Cache handles for O(1) state updates)
        self.ball_handle = self.server.scene.add_icosphere(
            "/ball",
            radius=self.ent_cfg.ball.r,
            color=b_c[:3]
        )
        
        self.r1_handle = self.server.scene.add_box(
            "/racket1",
            dimensions=(self.ent_cfg.racket.r * 2, self.ent_cfg.racket.r * 2, self.ent_cfg.racket.t),
            color=r1_c[:3]
        )
        
        self.r2_handle = self.server.scene.add_box(
            "/racket2",
            dimensions=(self.ent_cfg.racket.r * 2, self.ent_cfg.racket.r * 2, self.ent_cfg.racket.t),
            color=r2_c[:3]
        )

        if self.rnd_cfg.get("show_grid", True):
            self.server.scene.add_grid("/grid", width=10, height=10)

    def reset(self):
        pass

    def _euler_to_quat(self, rpy):
        # Scipy outputs [x, y, z, w]. Viser expects [w, x, y, z]
        q = Rotation.from_euler('xyz', rpy).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    def render(self, ball_x, racket1_x=None, racket2_x=None):
        # Update kinematic states instantly
        self.ball_handle.position = ball_x[0:3]
        self.ball_handle.wxyz = self._euler_to_quat(ball_x[3:6])
        
        if racket1_x is not None:
            self.r1_handle.position = racket1_x[0:3]
            self.r1_handle.wxyz = self._euler_to_quat(racket1_x[3:6])
            
        if racket2_x is not None:
            self.r2_handle.position = racket2_x[0:3]
            self.r2_handle.wxyz = self._euler_to_quat(racket2_x[3:6])