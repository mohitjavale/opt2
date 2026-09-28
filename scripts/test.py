import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import shared_memory
import numpy as np
from omegaconf import OmegaConf

# from opt2.renderer.meshcat_renderer import MeshcatRenderer as Renderer
from opt2.renderer.mujoco_renderer import MujocoRenderer as Renderer
from opt2.physics.numpy_physics import NumpyPhysics as Physics

# 12 (ball) + 6 (racket1) + 6 (racket2) = 24 total floats
STATE_SHAPE = (24,)
STATE_DTYPE = np.float64
STATE_BYTES = int(np.prod(STATE_SHAPE) * np.dtype(STATE_DTYPE).itemsize)

class SharedStateBuffer:
    def __init__(self, shm_name, lock):
        self.shm_name = shm_name
        self.lock = lock

    def __enter__(self):
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        self.array = np.ndarray(STATE_SHAPE, dtype=STATE_DTYPE, buffer=self.shm.buf)
        return self

    def __exit__(self, *args):
        self.shm.close()

    def write(self, data):
        with self.lock:
            self.array[:] = data[:]

    def read_dict(self):
        with self.lock:
            local_copy = self.array.copy()
            
        return {
            "ball_x": local_copy[0:12],
            "racket1_x": local_copy[12:18],
            "racket2_x": local_copy[18:24]
        }

def physics_worker(lock, shm_name, cfg):
    # Flatten the nested config and append constants for the physics engine
    params = {
        'table_l': cfg.entities.table.l,
        'table_w': cfg.entities.table.w,
        'ball_r': cfg.entities.ball.r,
        'racket_r': cfg.entities.racket.r,
        'g': np.array([0, 0, -9.81]),
        'C_D': 0.141, 'C_L': 0.001,
        'mu': 0.102, 'epsilon_t': 0.883,
        'k_pv': 0.703, 'k_pw': 2222.222, 'epsilon_r': 0.788
    }
    
    physics = Physics(params)
    
    dt = 0.002
    # Initial state: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz]
    ball_state = np.array([1.5, 0.5, 0.0, 0,0,0, 0,0,0, 0,0,0], dtype=STATE_DTYPE)
    r1 = np.array([-1.4, 0.0, 0.2, 0.0, np.pi/2, 0.0], dtype=STATE_DTYPE)
    r2 = np.array([1.4, 0.0, 0.2, 0.0, -np.pi/2, 0.0], dtype=STATE_DTYPE)
    
    local_buffer = np.zeros(STATE_SHAPE, dtype=STATE_DTYPE)
    local_buffer[12:18] = r1
    local_buffer[18:24] = r2
    
    with SharedStateBuffer(shm_name, lock) as state:
        while True:
            ball_state = physics.step(ball_state, r1, r2, dt)
            
            local_buffer[0:12] = ball_state
            state.write(local_buffer)
            
            # time.sleep(dt) 

def render_worker(lock, shm_name, cfg):
    renderer = Renderer(cfg.entities, cfg.renderer)
    print("MuJoCo Viewer active." if not hasattr(renderer, 'vis') else f"MeshCat URL: {renderer.vis.url()}")
        
    frames = 0
    last_time = time.time()
    
    with SharedStateBuffer(shm_name, lock) as state:
        while True:
            state_dict = state.read_dict()
            renderer.render(**state_dict)
            frames += 1
            
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"Max Renderer FPS: {frames}")
                frames = 0
                last_time = current_time

def main():
    cfg = OmegaConf.load("src/opt2/config/default.yaml")
    lock = mp.Lock()
    
    with SharedMemoryManager() as smm:
        raw_shm = smm.SharedMemory(size=STATE_BYTES)
        
        workers = [
            mp.Process(target=physics_worker, args=(lock, raw_shm.name, cfg)),
            mp.Process(target=render_worker, args=(lock, raw_shm.name, cfg))
        ]
        
        for w in workers: w.start()
        
        try:
            for w in workers: w.join()
        except KeyboardInterrupt:
            print("\nTerminating processes...")
            for w in workers: w.terminate()

if __name__ == "__main__":
    main()
