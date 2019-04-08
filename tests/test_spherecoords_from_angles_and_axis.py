"""3D backpropagation with tilted axis of rotation: sphere coordinates"""
import numpy as np

import odtbrain
import odtbrain._alg3d_bppt


def test_simple_sphere():
    """simple geometrical tests"""
    angles = np.array([0, np.pi/2, np.pi])
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]

    results = []

    for tilted_axis in axes:
        angle_coords = odtbrain._alg3d_bppt.sphere_points_from_angles_and_tilt(
            angles, tilted_axis)
        results.append(angle_coords)

    s2 = 1/np.sqrt(2)
    correct = np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                        [[0, 0, 1], [1, 0, 0], [0, 0, -1]],
                        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                        [[0, 0, 1], [s2, .5, .5], [0, 1, 0]],
                        [[s2, 0, s2], [s2, 0, s2], [s2, 0, s2]],
                        [[s2, 0, s2],
                         [0.87965281125489458, s2/3*2, 0.063156230327168605],
                         [s2/3, s2/3*4, s2/3]],
                        ])
    assert np.allclose(correct, np.array(results))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()

    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, _zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    axes = [[0, 1, 0], [0, 1, 0.1], [0, 1, -1], [1, 0.1, 0]]
    colors = ["k", "blue", "red", "green"]
    angles = np.linspace(0, 2*np.pi, 100)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(axes)):
        tilted_axis = axes[i]
        color = colors[i]
        tilted_axis = np.array(tilted_axis)
        tilted_axis = tilted_axis/np.sqrt(np.sum(tilted_axis**2))

        angle_coords = odtbrain._alg3d_bppt.sphere_points_from_angles_and_tilt(
            angles, tilted_axis)

        u, v, w = tilted_axis
        a = Arrow3D([0, u], [0, v], [0, w], mutation_scale=20,
                    lw=1, arrowstyle="-|>", color=color)
        ax.add_artist(a)
        ax.scatter(angle_coords[:, 0], angle_coords[:, 1],
                   angle_coords[:, 2], c=color, marker='o')

    radius = 1
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-radius*1.5, radius*1.5)
    ax.set_ylim(-radius*1.5, radius*1.5)
    ax.set_zlim(-radius*1.5, radius*1.5)
    plt.tight_layout()

    plt.show()
