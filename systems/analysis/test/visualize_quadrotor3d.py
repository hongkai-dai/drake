import numpy as np
import pickle

import pydrake
from pydrake.systems.framework import (
    DiagramBuilder,
)
from pydrake.examples import (
    QuadrotorGeometry,
    QuadrotorPlant,
)
from pydrake.geometry import (
    Box,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Rgba,
    StartMeshcat,
    SceneGraph,
)
from pydrake.math import (
    RigidTransform,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True


def main():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    scene_graph = builder.AddSystem(SceneGraph())
    num_snaps = 3
    quadrotors = [None] * num_snaps
    geoms = [None] * num_snaps
    for i in range(num_snaps):
        quadrotors[i] = builder.AddSystem(QuadrotorPlant())
        quadrotors[i].set_name(f"quadrotor{i}")
        geoms[i] = QuadrotorGeometry.AddToBuilder(
            builder, quadrotors[i].get_output_port(0), f"quadrotor_geo{i}", scene_graph)

    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception))

    meshcat.SetObject("ground", Box(10, 10, 0.05), rgba=Rgba(0.5, 0.5, 0.5))
    meshcat.SetTransform("ground", RigidTransform(np.array([0, 0, -0.175])))

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    quadrotor_contexts = [None] * num_snaps
    for i in range(num_snaps):
        quadrotor_contexts[i] = diagram.GetMutableSubsystemContext(quadrotors[i], diagram_context)

    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_sim_clf3_cbf21.pickle", "rb") as input_file:
        load_data = pickle.load(input_file)
        state_data = load_data["state_data"]
        clf_data = load_data["clf_data"]
        cbf_data = load_data["cbf_data"]
        time_data = load_data["time_data"]
    quadrotor_contexts[0].SetContinuousState(state_data[:, 0])
    quadrotor_contexts[1].SetContinuousState(state_data[:, 3000])
    quadrotor_contexts[2].SetContinuousState(state_data[:, -1])

    meshcat.SetLine("traj", state_data[:3, :], 1, Rgba(1, 0, 0, 0.5))

    diagram.Publish(diagram_context)
    print(f"{meshcat.ws_url()}")

    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_sim_clf3_cbf21_2.pickle", "rb") as input_file:
        load_data_clf = pickle.load(input_file)
        state_data_clf = load_data_clf["state_data"]
        time_data_clf = load_data_clf["time_data"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    z_clf_cbf_handle, = ax1.plot(time_data, state_data[2, :])
    z_clf_handle, = ax1.plot(time_data_clf, state_data_clf[2, :])
    ax1.plot(time_data, np.ones_like(time_data) * -0.15, '--', color='r')
    ax1.set_xlabel("time (s)")
    ax1.xaxis.label.set_fontsize(20)
    ax1.set_ylabel("z (m)")
    ax1.yaxis.label.set_fontsize(20)
    ax1.set_title("Quadrotor z height")
    ax1.title.set_fontsize(16)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.legend([z_clf_cbf_handle, z_clf_handle], ["CBF-CLF QP controller", "CLF QP controller"], fontsize=18)
    fig1.set_tight_layout(True)
    for fig_format in ["png", "pdf"]:
        fig1.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/quadrotor_clf_cbf_z_1." + fig_format, format=fig_format)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(time_data, clf_data.squeeze())
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("V(x(t))")
    ax2.set_title("CLF trajectory with CBF-CLF QP controller")
    ax2.xaxis.label.set_fontsize(20)
    ax2.yaxis.label.set_fontsize(20)
    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.title.set_fontsize(16)
    fig2.set_tight_layout(True)
    for fig_format in ["png", "pdf"]:
        fig2.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/quadrotor_clf_cbf_V_1." + fig_format, format=fig_format)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(time_data, cbf_data.squeeze())
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("h(x(t))")
    ax3.set_title("CBF trajectory with CBF-CLF QP controller")
    ax3.xaxis.label.set_fontsize(20)
    ax3.yaxis.label.set_fontsize(20)
    ax3.xaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)
    ax3.title.set_fontsize(16)
    fig3.set_tight_layout(True)
    for fig_format in ["png", "pdf"]:
        fig3.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/quadrotor_clf_cbf_h_1." + fig_format, format=fig_format)

if __name__ == "__main__":
    main()



