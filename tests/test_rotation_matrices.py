"""Test 3D backpropagation with tilted axis of ration: matrices"""
import numpy as np

import odtbrain
import odtbrain._alg3d_bppt


def test_rotate_points_to_axis():
    # rotation of axis itself always goes to y-axis
    rot1 = odtbrain._alg3d_bppt.rotate_points_to_axis(
        points=[[1, 2, 3]], axis=[1, 2, 3])
    assert rot1[0][0] < 1e-14
    assert rot1[0][2] < 1e-14
    rot2 = odtbrain._alg3d_bppt.rotate_points_to_axis(
        points=[[-3, .6, .1]], axis=[-3, .6, .1])
    assert rot2[0][0] < 1e-14
    assert rot2[0][2] < 1e-14

    sq2 = np.sqrt(2)
    # rotation to 45deg about x
    points = [[0, 0, 1], [1, 0, 0], [1, 1, 0]]
    rot3 = odtbrain._alg3d_bppt.rotate_points_to_axis(
        points=points, axis=[0, 1, 1])
    assert np.allclose(rot3[0], [0, 1/sq2, 1/sq2])
    assert np.allclose(rot3[1], [1, 0, 0])
    assert np.allclose(rot3[2], [1, 1/sq2, -1/sq2])

    # rotation to 45deg about y
    points = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]
    rot4 = odtbrain._alg3d_bppt.rotate_points_to_axis(
        points=points, axis=[1, 0, 1])
    assert np.allclose(rot4[0], [-.5, 1/sq2, .5])
    assert np.allclose(rot4[1], [.5, 1/sq2, -.5])
    assert np.allclose(rot4[2], [1/sq2, 0, 1/sq2])

    # Visualization
    # plt, Arrow3D = setup_mpl()
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    # for vec in points:
    #     u,v,w = vec
    #     a = Arrow3D([0,u],[0,v],[0,w], mutation_scale=20,
    #                 lw=1, arrowstyle="-|>", color="k")
    #     ax.add_artist(a)
    # for vec in rot4:
    #     u,v,w = vec
    #     a = Arrow3D([0,u],[0,v],[0,w], mutation_scale=20, lw=1,
    #                 arrowstyle="-|>", color="b")
    #     ax.add_artist(a)
    #  radius=1
    #  ax.set_xlabel('X')
    #  ax.set_ylabel('Y')
    #  ax.set_zlabel('Z')
    # ax.set_xlim(-radius*1.5, radius*1.5)
    # ax.set_ylim(-radius*1.5, radius*1.5)
    # ax.set_zlim(-radius*1.5, radius*1.5)
    # plt.tight_layout()
    # plt.show()

    # rotation to -90deg about z
    points = [[0, 0, 1], [1, 0, 1], [1, -1, 0]]
    rot4 = odtbrain._alg3d_bppt.rotate_points_to_axis(
        points=points, axis=[1, 0, 0])
    assert np.allclose(rot4[0], [0, 0, 1])
    assert np.allclose(rot4[1], [0, 1, 1])
    assert np.allclose(rot4[2], [1, 1, 0])

    # negative axes
    # In this case, everything is rotated in the y-z plane
    # (this case is not physical for tomogrpahy)
    points = [[0, 0, 1], [1, 0, 0], [1, -1, 0]]
    rot4 = odtbrain._alg3d_bppt.rotate_points_to_axis(
        points=points, axis=[0, -1, 0])
    assert np.allclose(rot4[0], [0, 0, -1])
    assert np.allclose(rot4[1], [1, 0, 0])
    assert np.allclose(rot4[2], [1, 1, 0])


def test_rotation_matrix_from_point():
    """
    `rotation_matrix_from_point` generates a matrix that rotates a point at
    [0,0,1] to the position of the argument of the method.
    """
    sq2 = np.sqrt(2)
    # identity
    m1 = odtbrain._alg3d_bppt.rotation_matrix_from_point([0, 0, 1])
    assert np.allclose(np.dot(m1, [1, 2, 3]), [1, 2, 3])
    assert np.allclose(np.dot(m1, [-3, .5, -.6]), [-3, .5, -.6])

    # simple
    m2 = odtbrain._alg3d_bppt.rotation_matrix_from_point([0, 1, 1])
    assert np.allclose(np.dot(m2, [0, 0, 1]), [0, 1/sq2, 1/sq2])
    assert np.allclose(np.dot(m2, [0, 1, 1]), [0, sq2, 0])
    assert np.allclose(np.dot(m2, [1, 0, 0]), [1, 0, 0])

    # negative
    m3 = odtbrain._alg3d_bppt.rotation_matrix_from_point([-1, 1, 0])
    assert np.allclose(np.dot(m3, [1, 0, 0]), [0, 0, -1])
    assert np.allclose(np.dot(m3, [0, 1, 1]), [0, sq2, 0])
    assert np.allclose(np.dot(m3, [0, -1/sq2, -1/sq2]), [0, -1, 0])
    assert np.allclose(np.dot(m3, [0, 1/sq2, -1/sq2]), [-1, 0, 0])
    assert np.allclose(np.dot(m3, [0, -1/sq2, 1/sq2]), [1, 0, 0])


def setup_mpl():
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa F01
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

    return plt, Arrow3D


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()

    import scipy.ndimage
    plt, Arrow3D = setup_mpl()

    # Testarray
    N = 50
    A = 41
    proj = np.zeros((N, N, N))
    proj[(N)/2, (N)/2, :(N)/2] = np.abs(np.linspace(-10, 1, (N)/2))

    # By default, the rotational axis in _Back_3D_tilted is the y-axis.
    # Define a rotational axis with a slight offset in x and in z.
    axis = np.array([0.0, 1, .1])
    axis /= np.sqrt(np.sum(axis**2))

    # Now, obtain the 3D angles that are equally distributed on the unit
    # sphere and correspond to the positions of projections that we would
    # measure.
    angles = np.linspace(0, 2*np.pi, A, endpoint=False)
    # The first point in that array will be in the x-z-plane.
    points = odtbrain._alg3d_bppt.sphere_points_from_angles_and_tilt(
        angles, axis)

    # The following steps are exactly those that are used in
    # odtbrain._alg3d_bppt.backpropagate_3d_tilted
    # to perform 3D reconstruction with tilted angles.
    u, v, w = axis
    theta = np.arccos(v)

    # We need three rotations.

    # IMPROTANT:
    # We perform the reconstruction such that the rotational axis
    # is equal to the y-axis! This is easier than implementing a
    # rotation about the rotational axis and tilting with theta
    # before and afterwards.

    # This is the rotation that tilts the projection in the
    # direction of the rotation axis (new y-axis).
    Rtilt = np.array([
        [1,             0,              0],
        [0, np.cos(theta),  np.sin(theta)],
        [0, -np.sin(theta),  np.cos(theta)],
    ])

    out = np.zeros((N, N, N))

    vectors = []

    for ang, pnt in zip(angles, points):
        Rcircle = np.array([
            [np.cos(ang),  0, np.sin(ang)],
            [0, 1,           0],
            [-np.sin(ang), 0, np.cos(ang)],
        ])

        DR = np.dot(Rtilt, Rcircle)

        # pnt are already rotated by R1
        vectors.append(np.dot(Rtilt, pnt))

        # We need to give this rotation the correct offset
        c = 0.5*np.array(proj.shape)
        offset = c-c.dot(DR.T)
        rotate = scipy.ndimage.interpolation.affine_transform(
            proj, DR, offset=offset,
            mode="constant", cval=0, order=2)
        proj *= 0.98
        out += rotate

    # visualize the axes
    out[0, 0, 0] = np.max(out)  # origin
    out[-1, 0, 0] = np.max(out)/2  # x
    out[0, -1, 0] = np.max(out)/3  # z
    # show arrows pointing at projection directions
    # (should form cone aligned with y)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for vec in vectors:
        u, v, w = vec
        a = Arrow3D([0, u], [0, v], [0, w],
                    mutation_scale=20, lw=1, arrowstyle="-|>")
        ax.add_artist(a)

    radius = 1
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-radius*1.5, radius*1.5)
    ax.set_ylim(-radius*1.5, radius*1.5)
    ax.set_zlim(-radius*1.5, radius*1.5)
    plt.tight_layout()
    plt.show()
