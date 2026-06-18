## Project Structure

```text
swarm-active-slam-capture/
├── AGENTS.md
├── documentation.md
├── README.md
├── requirements.txt
├── run_active_slam.py
├── shared_config.py
├── push.sh
├── DDFGO++/
│   ├── Animate_All_Agents_Mapping_Offscreen.py
│   ├── Animate_Mapping.py
│   ├── Animate_Single_Agent_Mapping_Offscreen.py
│   ├── config.py
│   ├── Custom_Factors.py
│   ├── Feature_Processing.py
│   ├── helper.py
│   ├── LandmarkRegistry.py
│   ├── Load_Target.py
│   ├── map_merging.py
│   ├── notify_helper.py
│   ├── Plot_Mapping_Results.py
│   ├── Plot_Telemetry_Func.py
│   ├── Recording.py
│   └── SwarmDDFGO++.py
├── SwarmCapture+/
│   ├── A_Convert2vhacd.py
│   ├── A_GUI_Interactive_Keypoint_Extraction.py
│   ├── A_Recording.py
│   ├── A_symbolic_potential_equations_derivatives_latex.py
│   ├── Animate_Mapping.py
│   ├── Animate_Simulation.py
│   ├── Animate_Simulation_Eye_In_Hand_View.py
│   ├── Animate_Simulation_Fixed_View.py
│   ├── Controllers.py
│   ├── Load_Target.py
│   ├── Neighborhood.py
│   ├── Observe_Target.py
│   ├── Plot_3D_Traj.py
│   ├── Plot_3D_Trajectory.py
│   ├── Plot_Telemetry.py
│   ├── Plot_Telemetry_Func.py
│   ├── Plot_Telemetry_Swarm.py
│   ├── PSO_Gain_Tuning.py
│   ├── Ray_Cast_Lidar.py
│   ├── Spacecraft_Swarm.py
│   ├── Swarm_Target_Capture+.py
│   └── Cube_Blender/
│       ├── Blue Back Face.png
│       ├── Cube.blend
│       ├── Cube.mtl
│       ├── Cube.obj
│       ├── Red Front Face.png
│       ├── Texture_Cube.png
│       ├── Texturing.pptx
│       └── Yellow Side Faces.png
├── simplified_2d/
│   ├── llm_swarms.md
│   ├── plan.md
│   ├── simplified_swarm.py
│   └── prototype2d/
│       ├── __init__.py
│       ├── animation.py
│       ├── config.json
│       ├── controllers.py
│       ├── delays.py
│       ├── io.py
│       ├── metrics.py
│       ├── model.py
│       ├── perception.py
│       ├── phases.md
│       ├── plotting.py
│       ├── simulator.py
│       ├── sketch.json
│       └── target_sketch_tk.py
└── utilities/
    ├── data/
    │   └── mock_data.py
    ├── contact_points/
    │   ├── candidate_gossip.py
    │   ├── contact_points.py
    │   ├── demo_candidate_gossip.py
    │   └── plane_ransac.py
    └── coverage/
        ├── coverage.py
        └── ellipsoid.py
```
