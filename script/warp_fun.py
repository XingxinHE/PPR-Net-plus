import torch
import warp as wp
import trimesh
import numpy as np

wp.init()
device = "cuda"

# --- 1. Load Surface Mesh via Trimesh ---
# Replace 'bunny.obj' with your file
# For this demo, we create a simple box if file doesn't exist
try:
    mesh = trimesh.load('/workspace/PPR-Net-plus/models/T.obj')
except:
    print("File not found, creating dummy box.")
    mesh = trimesh.creation.box(extents=[1, 1, 1])

# Extract data for Warp
mesh_vertices = wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3f, device=device)
mesh_indices = wp.array(np.array(mesh.faces.flatten(), dtype=np.int32), device=device)

# Create the Warp Mesh (Static BVH)
wp_mesh = wp.Mesh(points=mesh_vertices, indices=mesh_indices)

# --- 2. Create Target Point Cloud ---
# Let's pretend the 'real' object is at [1.0, 1.0, 1.0] with 45 deg rotation
# We want our mesh to move to this location to match these points.
target_transform = trimesh.transformations.compose_matrix(
    translate=[1.0, 1.0, 1.0], angles=[0, 0, 0.785]
)
target_points_np = trimesh.transform_points(mesh.sample(1000), target_transform)
target_points_np = target_points_np.astype(np.float32)

wp_target_points = wp.from_numpy(target_points_np, dtype=wp.vec3f, device=device)


# --- 3. The Inverse Logic Kernel ---
@wp.kernel
def compute_loss_inverse(
    mesh_id: wp.uint64,
    target_points: wp.array(dtype=wp.vec3f),
    pose_mesh_to_world: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32)
):
    tid = wp.tid()

    # 1. Construct the Transform (Mesh -> World)
    p = wp.vec3f(pose_mesh_to_world[0], pose_mesh_to_world[1], pose_mesh_to_world[2])
    q = wp.quaternion(pose_mesh_to_world[3], pose_mesh_to_world[4], pose_mesh_to_world[5], pose_mesh_to_world[6])

    xform = wp.transform(p, q)

    # 2. INVERSE: Map World Point -> Mesh Local Space
    world_pt = target_points[tid]

    # --- FIX START ---
    # Invert the transform
    inv_xform = wp.transform_inverse(xform)
    # Apply the inverted transform to the point
    local_pt = wp.transform_point(inv_xform, world_pt)
    # --- FIX END ---

    # 3. Query the static mesh in local space
    query = wp.mesh_query_point_no_sign(mesh_id, local_pt, 1.0e6)

    if query.result:
        pt_on_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)

        # Distance in local space
        dist = wp.length(pt_on_mesh - local_pt)

        # Accumulate loss
        wp.atomic_add(loss, 0, dist)

# --- 4. Optimization Loop ---

# Initial Guess: Identity pose (0,0,0) position, (0,0,0,1) rotation
# Format: [tx, ty, tz, qx, qy, qz, qw]
start_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         dtype=torch.float32, device=device, requires_grad=True)

optimizer = torch.optim.Adam([start_pose], lr=0.01)

print(f"Start Pose: {start_pose.detach().cpu().numpy()}")

for i in range(200):
    optimizer.zero_grad()

    tape = wp.Tape()
    with tape:
        wp_pose = wp.from_torch(start_pose, requires_grad=True)
        wp_loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        wp.launch(
            kernel=compute_loss_inverse,
            dim=target_points_np.shape[0],
            inputs=[wp_mesh.id, wp_target_points, wp_pose, wp_loss],
            outputs=[],
            device=device
        )

    tape.backward(wp_loss)

    # Transfer gradients
    if start_pose.grad is None:
        start_pose.grad = wp.to_torch(wp_pose.grad)
    else:
        start_pose.grad += wp.to_torch(wp_pose.grad)

    optimizer.step()

    # Normalize Quaternion (Important!)
    with torch.no_grad():
        start_pose[3:] = torch.nn.functional.normalize(start_pose[3:], dim=0)

    if i % 20 == 0:
        print(f"Iter {i}: Loss = {wp_loss.numpy()[0]:.4f}")

print("Optimization Finished.")
print(f"Final Estimated Pose (Pos): {start_pose[:3].detach().cpu().numpy()}")
# Expected: roughly [1.0, 1.0, 1.0] based on my target creation above