import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Ring:
    def __init__(self, radius, theta, phi):
        self.radius = radius
        self.theta = np.deg2rad(theta)
        self.phi = np.deg2rad(phi)
        self.rotated_coords = self.compute_rotated_coords()

    def compute_rotated_coords(self):
        t = np.linspace(0, 2 * np.pi, divs)
        x, y, z = self.radius * np.cos(t), self.radius * np.sin(t), np.zeros_like(t)

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(self.theta), -np.sin(self.theta)], [0, np.sin(self.theta), np.cos(self.theta)]])
        Rz = np.array([[np.cos(self.phi), -np.sin(self.phi), 0], [np.sin(self.phi), np.cos(self.phi), 0], [0, 0, 1]])
        return np.dot(Rz, np.dot(Rx, np.array([x, y, z])))

class Observer:
    def __init__(self, x, y, z):
        self.position = np.array([x, y, z])

class Particle:
    def __init__(self, velocity, start_index):
        self.velocity = velocity
        self.index = start_index
        self.positions = []
        self.total_distance = 0

    def move(self):
        self.index = (self.index + self.velocity) % divs
        self.total_distance += abs(self.velocity)

    def save_position(self, x, y, z):
        self.positions.append([x, y, z])

    def compute_observation_time(self, observer_position):
        path_length_per_segment = 2 * np.pi * ring.radius / divs
        t_travel = self.total_distance * path_length_per_segment
        x, y, z = self.positions[-1]
        distance_to_observer = np.sqrt((observer_position[0] - x)**2 +
                                       (observer_position[1] - y)**2 +
                                       (observer_position[2] - z)**2)
        t_light = distance_to_observer / c
        return t_travel + t_light

def frame_generator():
    frame = 0
    while True:
        yield frame
        frame += 1

def close_figure(event):
    plt.close(event.canvas.figure)

def update(frame):
    particle.move()
    x_coord = ring.rotated_coords[0, particle.index]
    y_coord = ring.rotated_coords[1, particle.index]
    z_coord = ring.rotated_coords[2, particle.index]

    particle.save_position(x_coord, y_coord, z_coord)
    t_tot = particle.compute_observation_time(observer.position)
    observation_times.append(t_tot)

    angle = 360 * (1 - particle.index / divs)
    angles.append(angle)

    print(f"At time {frame / 10:.1f} s, particle is at coordinates: [{x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f}]")

    particle_dot.set_data([x_coord], [y_coord])
    particle_dot.set_3d_properties([z_coord])

    time_text.set_text(f'Time: {frame / 10:.1f} s')
    return particle_dot, time_text,

# Main code
divs = 100
c = 0.5

ring = Ring(radius=5, theta=90, phi=90)
observer = Observer(x=0, y=0, z=50)
particle = Particle(velocity=-1, start_index=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ring.rotated_coords[0, :], ring.rotated_coords[1, :], ring.rotated_coords[2, :])
ax.scatter(observer.position[0], observer.position[1], observer.position[2], color='red', s=50, label='Observer')

ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=5, normalize=True)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=5, normalize=True)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=5, normalize=True)

ax.set_xlim([-7.5, 7.5])
ax.set_ylim([-7.5, 7.5])
ax.set_zlim([-7.5, 7.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

particle_dot, = ax.plot([], [], [], 'bo')
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

observation_times = []
angles = []

ani = FuncAnimation(fig, update, frames=297, interval=10, blit=True, repeat=False)
ani._stop = close_figure

plt.show()

min_observation_time = min(observation_times)
print(min_observation_time)
normalized_observation_times = [time - min_observation_time for time in observation_times]

fig, ax = plt.subplots(figsize=(9, 6))
plt.plot(normalized_observation_times, angles, 'o', markersize=3)
plt.xlabel('Relative Observation Time (seconds)')
plt.ylabel('Angular Position')
plt.grid(True)

ax.yaxis.set_major_locator(plt.MultipleLocator(30))

plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=85.5, color='r', linestyle='--')

plt.axhline(y=265, color='k', linestyle='--')
plt.axhline(y=100, color='k', linestyle='--')

plt.axvspan(0, 85.5, color='blue', alpha=0.1)

plt.text(127, 275, 'Image doubling angle', fontsize=12, va='center', ha='center')
plt.text(31, 110, 'Image annihilation angle', fontsize=12, va='center', ha='center')

plt.tight_layout()
ax.tick_params(axis="both", which="both", length=5, direction="in")

plt.show()

print("\nObservation Times (in seconds) for each position:")
for i, t in enumerate(observation_times):
    print(f"Angle {angles[i]:.2f} degrees: {t:.2f} seconds")
