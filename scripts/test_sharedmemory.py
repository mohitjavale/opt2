import time
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from omegaconf import OmegaConf

# Easily swap renderers by commenting/uncommenting:
from opt2.renderer.meshcat_renderer import MeshcatRenderer as Renderer
# from opt2.renderer.viser_renderer import ViserRenderer as Renderer
# from opt2.renderer.rerun_renderer import RerunRenderer as Renderer
# from opt2.renderer.mujoco_renderer import MujocoRenderer as Renderer

# Memory Layout Constants
# ball_x (6), racket1_x (6), racket2_x (6) = 18 elements total
STATE_SHAPE = (18,)
STATE_DTYPE = np.float64
STATE_BYTES = np.prod(STATE_SHAPE) * np.dtype(STATE_DTYPE).itemsize

def physics_worker(lock, shm_name, cfg):
    """Calculates state transitions and updates shared memory buffer."""
    # Attach to the allocated POSIX shared memory block
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # Map a NumPy array directly over the raw shared memory byte buffer
    shared_state = np.ndarray(STATE_SHAPE, dtype=STATE_DTYPE, buffer=shm.buf)
    
    t = 0.0
    r1 = np.array([-1.4, 0.0, 0.2, 0.0, np.pi/2, 0.0])
    r2 = np.array([1.4, 0.0, 0.2, 0.0, -np.pi/2, 0.0])
    
    # Pre-allocate a flat local buffer to prevent locking during calculations
    local_buffer = np.zeros(STATE_SHAPE, dtype=STATE_DTYPE)
    
    try:
        while True:
            # 1. Compute state values outside the lock
            ball_z = abs(0.3 * np.sin(t * 3)) + cfg.entities.ball.r
            ball_x = np.array([np.cos(t), 0.5 * np.sin(t * 0.5), ball_z, 0, 0, 0])
            
            # Pack state components into the single flat layout array
            local_buffer[0:6] = ball_x
            local_buffer[6:12] = r1
            local_buffer[12:18] = r2
            
            # 2. Acquire lock and copy chunk into shared memory quickly
            with lock:
                shared_state[:] = local_buffer[:]
                
            t += 0.002
            time.sleep(0.002)
            # No time.sleep() here to find max renderer performance limit
    finally:
        shm.close()

def render_worker(lock, shm_name, cfg):
    """Pulls states directly from shared memory and feeds the renderer framework."""
    renderer = Renderer(cfg.entities, cfg.renderer)
    
    if hasattr(renderer, 'vis'):
        print(f"MeshCat URL: {renderer.vis.url()}")
    else:
        print("MuJoCo Viewer active.")
        
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_state = np.ndarray(STATE_SHAPE, dtype=STATE_DTYPE, buffer=shm.buf)
    
    # Pre-allocated array for copying shared memory states locally
    local_state = np.zeros(STATE_SHAPE, dtype=STATE_DTYPE)
    
    frames = 0
    last_time = time.time()
    print("Starting shared memory benchmark... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # 1. Acquire lock and extract memory slice
            with lock:
                local_state[:] = shared_state[:]
            
            # 2. Reconstruct keyword dictionary arguments for the renderer out of the lock
            state = {
                "ball_x": local_state[0:6],
                "racket1_x": local_state[6:12],
                "racket2_x": local_state[12:18]
            }
            
            # 3. Render the frame execution pass
            renderer.render(**state)
            frames += 1
            
            # FPS Benchmark Statistics Printing Logic
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"Max Renderer FPS (Shared Memory): {frames}")
                frames = 0
                last_time = current_time
    finally:
        shm.close()

def main():
    cfg = OmegaConf.load("src/opt2/config/default.yaml")
    
    # Create the inter-process exclusion lock
    lock = mp.Lock()
    
    # Allocate block layout within Linux /dev/shm Virtual Memory System
    shm = shared_memory.SharedMemory(create=True, size=STATE_BYTES)
    shm_name = shm.name
    
    workers = [
        mp.Process(target=physics_worker, args=(lock, shm_name, cfg)),
        mp.Process(target=render_worker, args=(lock, shm_name, cfg))
    ]
    
    for w in workers: w.start()
    
    try:
        for w in workers: w.join()
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        for w in workers: w.terminate()
    finally:
        # Crucial clean up routines to avoid system-level RAM leak footprints
        shm.close()
        shm.unlink()
        print("Shared memory unlinked successfully.")

if __name__ == "__main__":
    main()
