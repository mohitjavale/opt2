import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation

class MujocoRenderer:
    def __init__(self, entities_cfg, render_cfg):
        self.ent_cfg = entities_cfg
        self.rnd_cfg = render_cfg
        
        self.trail_history = deque(maxlen=self.rnd_cfg.trail_len)
        self._setup_scene()

    def _setup_scene(self):
        # Note: MuJoCo box sizes are half-extents, so we divide your full dims by 2

        # get colors
        t_c = self.ent_cfg.table.color
        n_c = self.ent_cfg.net.color
        b_c = self.ent_cfg.ball.color
        r1_c = self.ent_cfg.racket.color1
        r2_c = self.ent_cfg.racket.color2

        xml = f"""
        <mujoco>
          <visual>
            <global offwidth="1920" offheight="1080"/>
          </visual>
          <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            
            <body name="table" pos="0 0 {-self.ent_cfg.table.h/2}">
              <geom type="box" size="{self.ent_cfg.table.l/2} {self.ent_cfg.table.w/2} {self.ent_cfg.table.h/2}" rgba="{t_c[0]} {t_c[1]} {t_c[2]} {t_c[3]}"/>
            </body>
            
            <body name="net" pos="0 0 {self.ent_cfg.net.h/2}">
              <geom type="box" size="{self.ent_cfg.net.l/2} {self.ent_cfg.net.w/2} {self.ent_cfg.net.h/2}" rgba="{n_c[0]} {n_c[1]} {n_c[2]} {n_c[3]}"/>
            </body>
            
            <body name="ball" mocap="true">
              <geom type="sphere" size="{self.ent_cfg.ball.r}" rgba="{b_c[0]} {b_c[1]} {b_c[2]} {b_c[3]}"/>
            </body>
            
            <body name="racket1" mocap="true">
              <geom type="box" size="{self.ent_cfg.racket.r} {self.ent_cfg.racket.r} {self.ent_cfg.racket.t/2}" rgba="{r1_c[0]} {r1_c[1]} {r1_c[2]} {r1_c[3]}"/>
            </body>
            <body name="racket2" mocap="true">
              <geom type="box" size="{self.ent_cfg.racket.r} {self.ent_cfg.racket.r} {self.ent_cfg.racket.t/2}" rgba="{r2_c[0]} {r2_c[1]} {r2_c[2]} {r2_c[3]}"/>
            </body>
          </worldbody>
        </mujoco>
        """
        
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        
        # Cache mocap IDs for fast updates
        self.mocap_ball = self.model.body("ball").mocapid[0]
        self.mocap_r1 = self.model.body("racket1").mocapid[0]
        self.mocap_r2 = self.model.body("racket2").mocapid[0]
        
        # Launch non-blocking passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)


        with self.viewer.lock():
            self.viewer.cam.azimuth = 180       # 90 degrees looks from +X towards -X
            self.viewer.cam.elevation = -15    # Look slightly downwards
            self.viewer.cam.distance = 4.0     # Distance from the center
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        
        if not self.rnd_cfg.get("show_grid", True):
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_GRID] = False

    def reset(self):
        self.trail_history.clear()
        if self.viewer.is_running():
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
            self.viewer.sync()

    def _euler_to_quat(self, rpy):
        # Scipy outputs [x, y, z, w]. MuJoCo expects [w, x, y, z].
        q = Rotation.from_euler('xyz', rpy).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    def render(self, ball_x, racket1_x=None, racket2_x=None):
        if not self.viewer.is_running():
            return

        # Update Mocap bodies
        self.data.mocap_pos[self.mocap_ball] = ball_x[0:3]
        self.data.mocap_quat[self.mocap_ball] = self._euler_to_quat(ball_x[3:6])
        
        if racket1_x is not None:
            self.data.mocap_pos[self.mocap_r1] = racket1_x[0:3]
            self.data.mocap_quat[self.mocap_r1] = self._euler_to_quat(racket1_x[3:6])
        if racket2_x is not None:
            self.data.mocap_pos[self.mocap_r2] = racket2_x[0:3]
            self.data.mocap_quat[self.mocap_r2] = self._euler_to_quat(racket2_x[3:6])

        # Apply kinematic state
        mujoco.mj_forward(self.model, self.data)

        self.trail_history.append(ball_x[0:3].copy())

        # Safely lock the viewer thread to prevent C-level memory corruption
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            
            for i in range(len(self.trail_history) - 1):
                # Prevent overflow of pre-allocated C arrays
                if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                    break
                    
                p1 = self.trail_history[i]
                p2 = self.trail_history[i+1]
                
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), np.array([1, 1, 1, 1])
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, p1, p2
                )
                
                self.viewer.user_scn.ngeom += 1

        self.viewer.sync()