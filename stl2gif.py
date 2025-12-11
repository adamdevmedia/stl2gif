import sys
import subprocess
import os
import math
import importlib.util
import tkinter as tk
from tkinter import filedialog
import time

# install/check libraries
def ensure(lib, import_name=None):
    if import_name is None:
        import_name = lib
    if importlib.util.find_spec(import_name) is not None:
        print(f"{lib} ok")
        return
    print(f"{lib} missing, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", lib], check=True)

# ensure all required packages
for lib, import_name in [("numpy", None), ("trimesh", None), ("pyrender", None), ("imageio", None), ("Pillow", "PIL"), ("pyfqmr", None)]:
    ensure(lib, import_name)

# imports after install
import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image, ImageDraw, ImageFont

def pick_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="select stl",
        filetypes=[("stl files", "*.stl")]
    )

def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path], check=True)
        else:  # linux
            subprocess.run(["xdg-open", path], check=True)
        return True
    except Exception as e:
        print(f"Could not open file: {e}")
        return False

def make_rotating_gif(stl_path, duration_seconds=15, fps=20):
    mesh = trimesh.load(stl_path)
    
    # Simplify mesh if it has too many faces (speeds up rendering significantly)
    if len(mesh.faces) > 50000:
        print(f"Simplifying mesh from {len(mesh.faces)} faces...")
        try:
            mesh = mesh.simplify_quadric_decimation(50000)
            print(f"Simplified to {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Quadric decimation failed, trying alternate method...")
            try:
                # Fallback to vertex clustering
                mesh = mesh.simplify_vertex_clustering(voxel_size=mesh.extents.max() / 100)
                print(f"Simplified to {len(mesh.faces)} faces using vertex clustering")
            except Exception as e2:
                print(f"Simplification not available, continuing with full mesh (may be slower)")
    
    mesh.merge_vertices(0.2)
    trimesh.repair.fix_normals(mesh)
    
    # Create output path in same directory with same name
    output_path = os.path.splitext(stl_path)[0] + ".gif"
    
    frames = int(duration_seconds * fps)
    tmp_dir = "spin_frames"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Calculate model dimensions and center
    bounds = mesh.bounds
    centroid = mesh.centroid
    size = bounds[1] - bounds[0]
    max_dim = max(size)
    max_horizontal = max(size[0], size[1])  # X and Y
    vertical_dim = size[2]  # Z dimension
    
    # For isometric view, camera should be at 45° horizontally and ~35.26° vertically
    iso_angle = math.radians(35.264)  # arctan(1/sqrt(2)) for true isometric
    
    # Calculate distance so model takes up ~52.5% of view (75% * 0.7 for more zoom out)
    fov = np.pi / 3.0
    target_coverage = 0.75 * 0.7
    
    # Base distance uses max horizontal dimension for vertical rotation
    base_cam_distance = max_horizontal / (2 * target_coverage * math.tan(fov / 2))
    
    # For horizontal rotation, we need to account for vertical dimension
    # When looking from the side, we see the vertical dimension
    vertical_cam_distance = vertical_dim / (2 * target_coverage * math.tan(fov / 2))
    
    # Use the larger of the two to ensure model fits in both rotations
    phase1_cam_distance = max(base_cam_distance, vertical_cam_distance * 0.8)
    phase2_cam_distance = max(base_cam_distance, vertical_cam_distance * 0.95)  # Less zoomed out
    
    # Position camera for fixed isometric view (use phase1 distance initially)
    cam_height = phase1_cam_distance * math.sin(iso_angle)
    cam_horizontal = phase1_cam_distance * math.cos(iso_angle)
    cam_pos_phase1 = centroid + np.array([cam_horizontal, cam_horizontal, cam_height])
    
    # For phase 2, adjust distance
    cam_height_phase2 = phase2_cam_distance * math.sin(iso_angle)
    cam_horizontal_phase2 = phase2_cam_distance * math.cos(iso_angle)
    cam_pos_phase2 = centroid + np.array([cam_horizontal_phase2, cam_horizontal_phase2, cam_height_phase2])
    
    # Create initial camera pose
    cam_pos = cam_pos_phase1
    forward = centroid - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, np.array([0, 0, 1]))
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos
    
    # Animation phases:
    # Phase 1: 3 rotations around vertical (Z) axis while zooming out 10%
    # Transition: 1 second smooth camera change
    # Phase 2: 3 rotations around horizontal (X) axis
    # Transition: 1 second smooth camera change back
    # Phase 3: Return rotations to start while zooming in 10%
    
    transition_frames = int(fps * 0.5)  # 1 second for smooth transition between phases
    
    # Split remaining frames into 3 equal phases
    remaining = frames - (transition_frames * 2)
    phase_frames = remaining // 3
    
    intro_frames = phase_frames  # Phase 1 with zoom out
    phase1_frames = 0  # No separate phase 1, it's part of intro
    phase2_frames = phase_frames  # Phase 2
    outro_frames = remaining - (phase_frames * 2)  # Phase 3 with zoom in
    
    # scene setup
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])  # Add ambient light to reduce directional light needs
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    
    # Create camera (fixed position)
    camera = pyrender.PerspectiveCamera(yfov=fov)
    cam_node = scene.add(camera, pose=cam_pose)
    
    # Single lighter directional light (ambient handles the rest) at camera position
    light = pyrender.DirectionalLight(intensity=2.0)
    light_node = scene.add(light, pose=cam_pose)
    
    renderer = pyrender.OffscreenRenderer(1024, 1024)
    
    # Get model dimensions for display
    dim_x, dim_y, dim_z = size
    
    # Pre-load font once
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
    
    print(f"Rendering {frames} frames...")
    for i in range(frames):
        # Determine which camera position to use and interpolate if transitioning
        if i < intro_frames:
            # Phase 1: zoom out while rotating
            cam_pos = cam_pos_phase1
        elif i < intro_frames + transition_frames:
            # Transition 1: smoothly interpolate camera position from phase1 to phase2
            transition_progress = (i - intro_frames) / transition_frames
            cam_pos = cam_pos_phase1 + (cam_pos_phase2 - cam_pos_phase1) * transition_progress
        elif i < intro_frames + transition_frames + phase2_frames:
            # Phase 2: horizontal rotation
            cam_pos = cam_pos_phase2
        elif i < intro_frames + transition_frames + phase2_frames + transition_frames:
            # Transition 2: smoothly interpolate camera position from phase2 back to phase1
            transition_progress = (i - intro_frames - transition_frames - phase2_frames) / transition_frames
            cam_pos = cam_pos_phase2 + (cam_pos_phase1 - cam_pos_phase2) * transition_progress
        else:
            # Phase 3: outro with zoom in
            cam_pos = cam_pos_phase1
        
        # Update camera look-at for current position
        forward = centroid - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos
        
        scene.set_pose(cam_node, cam_pose)
        scene.set_pose(light_node, cam_pose)
        
        # Determine scale factor for zoom effect - happens during rotation now
        if i < intro_frames:
            # Phase 1: zoom out from 90% to 100% while rotating
            progress = i / intro_frames
            scale = 0.9 + (0.1 * progress)
        elif i >= frames - outro_frames:
            # Phase 3: zoom in from 100% to 90% while rotating
            progress = (i - (frames - outro_frames)) / outro_frames
            scale = 1.0 - (0.1 * progress)
        else:
            # Middle: normal scale
            scale = 1.0
        
        # Determine which rotation phase we're in
        if i < intro_frames:
            # Phase 1: Rotate around vertical (Z) axis while zooming
            phase_progress = i / intro_frames
            angle = 2 * math.pi * phase_progress * 0.75  # 0.75 rotations
            
            rotation_matrix = np.array([
                [math.cos(angle), -math.sin(angle), 0, 0],
                [math.sin(angle),  math.cos(angle), 0, 0],
                [0,                0,               1, 0],
                [0,                0,               0, 1]
            ])
        elif i < intro_frames + transition_frames:
            # Transition 1: hold at final angle of phase 1 while camera moves
            angle = 2 * math.pi * 0.75  # Final angle from phase 1
            
            rotation_matrix = np.array([
                [math.cos(angle), -math.sin(angle), 0, 0],
                [math.sin(angle),  math.cos(angle), 0, 0],
                [0,                0,               1, 0],
                [0,                0,               0, 1]
            ])
        elif i < intro_frames + transition_frames + phase2_frames:
            # Phase 2: Rotate around horizontal (X) axis
            phase_progress = (i - intro_frames - transition_frames) / phase2_frames
            angle = 2 * math.pi * phase_progress * 0.75  # 0.75 rotations
            
            # Start from where phase 1 ended
            z_angle = 2 * math.pi * 0.75  # Final angle from phase 1
            z_rotation = np.array([
                [math.cos(z_angle), -math.sin(z_angle), 0, 0],
                [math.sin(z_angle),  math.cos(z_angle), 0, 0],
                [0,                  0,                 1, 0],
                [0,                  0,                 0, 1]
            ])
            
            x_rotation = np.array([
                [1, 0,                0,               0],
                [0, math.cos(angle), -math.sin(angle), 0],
                [0, math.sin(angle),  math.cos(angle), 0],
                [0, 0,                0,               1]
            ])
            
            rotation_matrix = z_rotation @ x_rotation
        elif i < intro_frames + transition_frames + phase2_frames + transition_frames:
            # Transition 2: hold at final angles while camera moves back
            z_angle = 2 * math.pi * 0.75
            x_angle = 2 * math.pi * 0.75
            
            z_rotation = np.array([
                [math.cos(z_angle), -math.sin(z_angle), 0, 0],
                [math.sin(z_angle),  math.cos(z_angle), 0, 0],
                [0,                  0,                 1, 0],
                [0,                  0,                 0, 1]
            ])
            
            x_rotation = np.array([
                [1, 0,                0,               0],
                [0, math.cos(x_angle), -math.sin(x_angle), 0],
                [0, math.sin(x_angle),  math.cos(x_angle), 0],
                [0, 0,                0,               1]
            ])
            
            rotation_matrix = z_rotation @ x_rotation
        else:
            # Phase 3: Return to starting position while zooming in
            z_angle = 2 * math.pi * 0.75  # Final Z angle
            x_angle = 2 * math.pi * 0.75  # Final X angle
            
            # Gradually return to identity
            outro_progress = (i - (frames - outro_frames)) / outro_frames
            
            # Reverse the rotations back to zero
            current_z = z_angle * (1 - outro_progress)
            current_x = x_angle * (1 - outro_progress)
            
            z_rotation = np.array([
                [math.cos(current_z), -math.sin(current_z), 0, 0],
                [math.sin(current_z),  math.cos(current_z), 0, 0],
                [0,                    0,                   1, 0],
                [0,                    0,                   0, 1]
            ])
            
            x_rotation = np.array([
                [1, 0,                    0,                   0],
                [0, math.cos(current_x), -math.sin(current_x), 0],
                [0, math.sin(current_x),  math.cos(current_x), 0],
                [0, 0,                    0,                   1]
            ])
            
            rotation_matrix = z_rotation @ x_rotation
        
        # Apply scale for zoom effect (scale the model, not the camera)
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale
        scale_matrix[1, 1] = scale
        scale_matrix[2, 2] = scale
        
        # Create translation matrices to rotate around centroid instead of origin
        # Translate to origin, rotate, then translate back
        translate_to_origin = np.eye(4)
        translate_to_origin[:3, 3] = -centroid
        
        translate_back = np.eye(4)
        translate_back[:3, 3] = centroid
        
        # Combine transformations: translate to origin, scale, rotate, translate back
        final_transform = translate_back @ rotation_matrix @ scale_matrix @ translate_to_origin
        scene.set_pose(mesh_node, final_transform)
        
        color, _ = renderer.render(scene)
        
        # Add dimensions text overlay
        img = Image.fromarray(color)
        draw = ImageDraw.Draw(img)
        
        # Format dimensions
        text_lines = [
            f"X: {dim_x:.2f}",
            f"Y: {dim_y:.2f}",
            f"Z: {dim_z:.2f}"
        ]
        
        # Position in top right with padding
        padding = 15
        line_spacing = 22
        
        for idx, line in enumerate(text_lines):
            # Get text size
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            
            x = img.width - text_width - padding
            y = padding + (idx * line_spacing)
            
            # Draw text with slight shadow for readability
            draw.text((x+1, y+1), line, fill=(0, 0, 0, 180), font=font)
            draw.text((x, y), line, fill=(255, 255, 255, 255), font=font)
        
        img.save(f"{tmp_dir}/f_{i:04d}.png")
        
        # Progress bar
        percent = ((i + 1) / frames) * 100
        bar_width = 40
        filled = int(bar_width * (i + 1) / frames)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\rRendering: [{bar}] {percent:.1f}%", end='', flush=True)
    
    print()  # New line after progress bar
    
    print("Compiling GIF...")
    # Use pillow mode for faster compilation with optimization
    imgs = []
    for i in range(frames):
        img = Image.open(f"{tmp_dir}/f_{i:04d}.png")
        imgs.append(img)
        
        # Progress bar for loading images
        percent = ((i + 1) / frames) * 100
        bar_width = 40
        filled = int(bar_width * (i + 1) / frames)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\rLoading frames: [{bar}] {percent:.1f}%", end='', flush=True)
    
    print()  # New line
    print("Saving GIF (this may take a moment)...")
    
    start_time = time.time()
    
    # Save with pillow directly - much faster for large GIFs
    imgs[0].save(
        output_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000/fps),  # duration in milliseconds
        loop=0,
        optimize=False  # Skip optimization for speed
    )
    
    elapsed = time.time() - start_time
    print(f"GIF saved in {elapsed:.1f} seconds")
    
    print(f"GIF saved to: {output_path}")
    
    # Try to open the file with better error handling
    if open_file(output_path):
        print("GIF opened successfully!")
    else:
        print(f"Please open manually: {output_path}")

if __name__ == "__main__":
    f = pick_file()
    if f:
        make_rotating_gif(f)
