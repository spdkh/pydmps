"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

import pydmps.dmp_discrete
from mpl_toolkits.mplot3d import Axes3D
import pytransform3d.rotations as pr
random_state = np.random.RandomState(42)

beta = 20.0 / np.pi
gamma = 100
R_halfpi = np.array(
    [
        [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
        [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
    ]
)
num_obstacles = 5
n_points = 20
x = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
pnts = np.array([[x_i, 2*x_i, x_i**2] for x_i in x])
# pnts = np.array([[np.cos(theta), np.sin(theta), 0] for theta in x[:-1]])
print(np.shape(pnts))

goals = pnts[::2]
obstacle_ids = np.random.choice(len(pnts)//2, num_obstacles, replace=False)
print('Obstacle IDs:', obstacle_ids)
obstacles = pnts[2 * obstacle_ids + 1]
# obstacles = np.random.random((num_obstacles, 2)) * 2 - 1

print('Obstacles:', obstacles)

def avoid_obstacles(y, dy, goal):
    p = np.zeros(3)

    for obstacle in obstacles:
        # based on (Hoffmann, 2009)

        # if we're moving
        # if np.linalg.norm(dy) > 1e-5:

            # get the angle we're heading in
            phi_dy = -np.arctan2(dy[1], dy[0])
            R_dy = np.array(
                [[np.cos(phi_dy), -np.sin(phi_dy)], [np.sin(phi_dy), np.cos(phi_dy)]]
            )
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0
            else:
                # calculate the angle of obj relative to the direction we're going
                phi = np.arctan2(obj_vec[1], obj_vec[0])

                dphi = gamma * phi * np.exp(-beta * abs(phi))
                R = np.dot(R_halfpi, np.outer(obstacle - y, dy))
                pval = -np.nan_to_num(np.dot(R, dy) * dphi)

                print(p, pval, p+pval)
            p += pval
            
    return p

EPSILON = 1e-10

def obstacle_avoidance_acceleration_3d(
        obstacles, y, yd, goal, gamma=1000.0, beta=20.0 / np.pi):
    """Compute acceleration for obstacle avoidance in 3D.

    Parameters
    ----------
    y : array, shape (..., 3)
        Current position(s).

    yd : array, shape (..., 3)
        Current velocity / velocities.

    obstacle_position : array, shape (3,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Obstacle avoidance parameter.

    beta : float, optional (default: 20 / pi)
        Obstacle avoidance parameter.

    Returns
    -------
    cdd : array, shape (..., 3)
        Accelerations.
    """
    p = np.zeros(3)

    for obstacle_position in obstacles:
        obstacle_diff = obstacle_position - y
        if np.linalg.norm(obstacle_diff) > np.linalg.norm(goal - y):
            cdd = 0
        else:
            r = 0.5 * np.pi * pr.norm_vector(np.cross(obstacle_diff, yd))
            R = pr.matrix_from_compact_axis_angle(r)
            theta = np.arccos(
                np.dot(obstacle_diff, yd)
                / (np.linalg.norm(obstacle_diff) * np.linalg.norm(yd) + EPSILON))
            cdd = gamma * np.dot(R, yd) * theta * np.exp(-beta * theta)
        p += cdd
    return p

n_dms = len(goals[0])
n_bfs = 10

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=n_dms, n_bfs=n_bfs, w=np.zeros((n_dms, n_bfs)))

y_des = dmp.imitate_path(goals.T)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Scatter plot for the first set of points
ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='r', marker='x', label='Obstacles', alpha=0.8, s = 100)

# Scatter plot for the second set of points
ax.plot(goals[:, 0], goals[:, 1], goals[:, 2], 'c--', label='Desired Path', lw=2, alpha=0.7)

y_track, dy_track, ddy_track = dmp.rollout(external_force=obstacle_avoidance_acceleration_3d(obstacles, dmp.y, dmp.dy, dmp.goal))
ax.plot(y_track[:, 0], y_track[:, 1], y_track[:, 2], "g--", label='DMP Imitated Original Path', lw=2, alpha=0.5)


# Add a legend
all_y = [[0, 0, 0]]


# run while moving the target up and to the right
y_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    y, _, _ = dmp.step()
    y_track.append(np.copy(y))
    # move the target slightly every time step
    # dmp.goal += np.array([-1e-3, 1e-2, 1e-3])
y_track = np.array(y_track)

# dmp.y = 0.3 * random_state.randn(3)
# y_track[0, :] = dmp.y
# print('Initial state:', dmp.y)
    
    
#     for t in range(dmp.timesteps):
#         y_track[t], dy_track[t], ddy_track[t] = dmp.step(
#             external_force=obstacle_avoidance_acceleration_3d(dmp.y, dmp.dy, goal)
#         )
#     all_y = np.vstack((all_y, y_track))

# ax.plot(all_y[1:, 0], all_y[1:, 1], all_y[1:, 2], c=color, lw=2, alpha=0.9)

# ax.plot(y_track[:, 0], y_track[:, 1], "b--", label='DMP path', lw=2, alpha=0.7)
plt.title("3D DMP system - obstacle avoidance")
# Set labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')


# plt.axis("equal")
# plt.xlim([-1.1, 1.1])
# plt.ylim([-1.1, 1.1])
plt.legend()
plt.show()
