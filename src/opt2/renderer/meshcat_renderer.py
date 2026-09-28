import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from collections import deque

class MeshcatRenderer:
    def __init__(self, entities_cfg, render_cfg):
        self.ent_cfg = entities_cfg
        self.rnd_cfg = render_cfg
        
        self.vis = meshcat.Visualizer()
        self._setup_scene()

    def _setup_scene(self):
        """Construct static geometries and assign colors."""

        # Disable Grid
        self.vis["/Grid"].set_property("visible", self.rnd_cfg.show_grid)


        # Table
        self.vis['table'].set_object(g.Box([self.ent_cfg.table.l, self.ent_cfg.table.w, self.ent_cfg.table.h]))
        self.vis['table'].set_transform(tf.translation_matrix([0, 0, -self.ent_cfg.table.h/2]))
        self.vis['table'].set_property("color", list(self.ent_cfg.table.color))
        
        # Net
        self.vis['net'].set_object(g.Box([self.ent_cfg.net.l, self.ent_cfg.net.w, self.ent_cfg.net.h]))
        self.vis['net'].set_transform(tf.translation_matrix([0, 0, self.ent_cfg.net.h/2]))
        self.vis['net'].set_property("color", list(self.ent_cfg.net.color))
        
        # Ball
        self.vis['ball'].set_object(g.Sphere(self.ent_cfg.ball.r))
        self.vis['ball'].set_property("color", list(self.ent_cfg.ball.color))
        
        # Rackets
        self.vis['racket1'].set_object(g.Box([self.ent_cfg.racket.r*2, self.ent_cfg.racket.r*2, self.ent_cfg.racket.t]))
        self.vis['racket1'].set_property("color", list(self.ent_cfg.racket.color1))
        
        self.vis['racket2'].set_object(g.Box([self.ent_cfg.racket.r*2, self.ent_cfg.racket.r*2, self.ent_cfg.racket.t]))
        self.vis['racket2'].set_property("color", list(self.ent_cfg.racket.color2))

        # Trail
        self.trail_history = deque(maxlen=self.rnd_cfg.trail_len)
        self.vis['trail'].set_object(g.Line(g.PointsGeometry(np.array([]).T)))

        # Camera
        # cam_pos = tf.translation_matrix([3.5, 0.0, 0.5])
        # cam_rot = tf.euler_matrix(0, -np.pi/2, 0)        
        # self.vis["/Cameras/default"].set_transform(cam_pos @ cam_rot)
        
    def reset(self):
        self.trail_history.clear()
        self.vis['trail'].set_object(g.Line(g.PointsGeometry(np.array([]).T)))

    def render(self, ball_x, racket1_x=None, racket2_x=None):

        # Ball
        self.vis['ball'].set_transform(tf.translation_matrix(ball_x[0:3]) @ tf.euler_matrix(*ball_x[3:6]))
            
        # Rackets
        if racket1_x is not None:
            self.vis['racket1'].set_transform(tf.translation_matrix(racket1_x[0:3]) @ tf.euler_matrix(*racket1_x[3:6]))
        if racket2_x is not None:
            self.vis['racket2'].set_transform(tf.translation_matrix(racket2_x[0:3]) @ tf.euler_matrix(*racket2_x[3:6]))

        # Trail line
        self.trail_history.append(ball_x[0:3])
        self.vis['trail'].set_object(g.Line(g.PointsGeometry(np.array(self.trail_history).T)))


    def close(self):
        self.vis.delete()
