import numpy as np
import rerun as rr
from collections import deque
from scipy.spatial.transform import Rotation

class RerunRenderer:
    def __init__(self, entities_cfg, render_cfg):
        self.ent_cfg = entities_cfg
        self.rnd_cfg = render_cfg
        
        # You can safely bring the trail back! Rerun handles massive arrays effortlessly.
        self.trail_history = deque(maxlen=self.rnd_cfg.trail_len)

        # Initialize and spawn the native Rust viewer process
        rr.init("table_tennis")
        rr.spawn() 

        # Set world coordinate system (Assuming Z-Up based on previous physics math)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._setup_scene()

    def _setup_scene(self):
        t_c = self.ent_cfg.table.color
        n_c = self.ent_cfg.net.color
        r1_c = self.ent_cfg.racket.color1
        r2_c = self.ent_cfg.racket.color2

        # 1. Log Static Environment
        rr.log(
            "world/table",
            rr.Boxes3D(
                half_sizes=[[self.ent_cfg.table.l / 2, self.ent_cfg.table.w / 2, self.ent_cfg.table.h / 2]],
                centers=[[0, 0, -self.ent_cfg.table.h / 2]],
                colors=[t_c[:3]]
            ),
            static=True
        )

        rr.log(
            "world/net",
            rr.Boxes3D(
                half_sizes=[[self.ent_cfg.net.l / 2, self.ent_cfg.net.w / 2, self.ent_cfg.net.h / 2]],
                centers=[[0, 0, self.ent_cfg.net.h / 2]],
                colors=[n_c[:3]]
            ),
            static=True
        )

        # 2. Log Racket Geometries at the origin (We will move them using Transforms later)
        rr.log(
            "world/rackets/r1/mesh",
            rr.Boxes3D(
                half_sizes=[[self.ent_cfg.racket.r, self.ent_cfg.racket.r, self.ent_cfg.racket.t / 2]],
                colors=[r1_c[:3]]
            ),
            static=True
        )
        
        rr.log(
            "world/rackets/r2/mesh",
            rr.Boxes3D(
                half_sizes=[[self.ent_cfg.racket.r, self.ent_cfg.racket.r, self.ent_cfg.racket.t / 2]],
                colors=[r2_c[:3]]
            ),
            static=True
        )

    def reset(self):
        self.trail_history.clear()

    def render(self, ball_x, racket1_x=None, racket2_x=None):
        # 1. Update Ball Position
        b_c = self.ent_cfg.ball.color
        rr.log(
            "world/ball",
            rr.Points3D(
                positions=[ball_x[0:3]],
                colors=[b_c[:3]],
                radii=[self.ent_cfg.ball.r]
            )
        )

        # 2. Update Trail 
        self.trail_history.append(ball_x[0:3].copy())
        if len(self.trail_history) > 1:
            rr.log(
                "world/trail",
                rr.LineStrips3D(
                    strips=[np.array(self.trail_history)],
                    colors=[b_c[:3]],
                    radii=[0.005]
                )
            )

        # 3. Update Racket Transforms (Orientation and Translation)
        if racket1_x is not None:
            # Scipy Rotation.as_quat() outputs [x, y, z, w], which matches Rerun's xyzw expected format
            q1 = Rotation.from_euler('xyz', racket1_x[3:6]).as_quat() 
            rr.log(
                "world/rackets/r1",
                rr.Transform3D(
                    translation=racket1_x[0:3],
                    rotation=rr.Quaternion(xyzw=q1)
                )
            )

        if racket2_x is not None:
            q2 = Rotation.from_euler('xyz', racket2_x[3:6]).as_quat()
            rr.log(
                "world/rackets/r2",
                rr.Transform3D(
                    translation=racket2_x[0:3],
                    rotation=rr.Quaternion(xyzw=q2)
                )
            )