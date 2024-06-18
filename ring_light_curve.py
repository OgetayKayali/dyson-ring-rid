import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


class Ring:
    def __init__(self, radius, theta, phi):  # Radius in light-seconds
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
        distance_to_observer = np.sqrt((observer_position[0] - x) ** 2 +
                                       (observer_position[1] - y) ** 2 +
                                       (observer_position[2] - z) ** 2)
        t_light = distance_to_observer / c
        return t_travel + t_light


# Main loop replacing the animation...
divs = 20000  # Divisions on the ring
c = 0.1  # Speed of light (fixing the particle's speed to 1)
rotations = 10  # How many rotations will be shown
start_index = 5000  # Define starting index for the particle

# Create objects
ring = Ring(radius=5, theta=90, phi=90)
observer = Observer(x=0, y=0, z=1000)
particle = Particle(velocity=-1, start_index=start_index)

# Initialize lists
observation_times = []
angles = []

# Calculate the initial angle from start_index
initial_angle = 360 * start_index / divs

# Calculate positions and observation times
for frame in range(rotations * divs):
    particle.move()
    x_coord = ring.rotated_coords[0, particle.index]
    y_coord = ring.rotated_coords[1, particle.index]
    z_coord = ring.rotated_coords[2, particle.index]

    particle.save_position(x_coord, y_coord, z_coord)
    t_tot = particle.compute_observation_time(observer.position)
    observation_times.append(t_tot)

    angle = initial_angle + 360 * frame / divs
    angles.append(angle)

# Normalize observation_times
min_observation_time = min(observation_times)
normalized_observation_times = [time - min_observation_time for time in observation_times]

# Prepare the main dataframe
df_main = pd.DataFrame(data={'Observation Time (seconds)': normalized_observation_times, 'Angle (degrees)': angles})

# Define the intervals
intervals = [0.1, 1, 5, 20]

# Create a figure and a set of subplots using gridspec
plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 2.7])

# Define the shaded area boundaries
x_start = 75
x_end = 325

# Plot observation_time vs angles in the first column
ax0 = plt.subplot(gs[:, 0])
ax0.plot(normalized_observation_times, angles, 'o', markersize=1)
ax0.set_xlabel('Relative Observation Time (seconds)')
ax0.set_ylabel('Number of Rotations')
ax0.grid(True, which='major')  # Display only major grid lines

# Calculate the range and set y-ticks at multiples of 360
y_max = max(angles)
tick_values = np.arange(0, y_max, 360)  # Steps of 360
tick_labels = [str(int(value / 360)) for value in tick_values]  # Convert angle to rotation count
ax0.set_yticks(tick_values)
ax0.set_yticklabels(tick_labels)  # Setting custom labels to represent the number of rotations

# Add dashed lines and shaded area for the left plot
ax0.axvline(x=x_start, color='k', linestyle='--')
ax0.axvline(x=x_end, color='k', linestyle='--')
ax0.axvspan(x_start, x_end, color='red', alpha=0.1)

ax0.tick_params(axis="both", which="both", length=5, direction="in")

# Create the four interval plots in the second column
for i, interval in enumerate(intervals):
    ax = plt.subplot(gs[i, 1])
    df = df_main.copy()
    df['Time Interval'] = (df['Observation Time (seconds)'] // interval * interval).astype(float)
    df['Cosine of Angle'] = np.abs(np.cos(np.deg2rad(df['Angle (degrees)'])))
    sum_cosine = df.groupby('Time Interval')['Cosine of Angle'].sum().reset_index()
    sum_cosine.rename(columns={'Cosine of Angle': 'Relative Brightness'}, inplace=True)
    ax.plot(sum_cosine['Time Interval'], sum_cosine['Relative Brightness'], marker='o', linestyle='-', markersize=1)
    ax.set_xlim(-15, 415)
    ax.set_title(f'Exposure Time: {interval} seconds')

    # Set x-label only for the bottom plot
    if i == len(intervals) - 1:
        ax.set_xlabel('Relative Observation Time')
    else:
        ax.tick_params(labelbottom=False)  # Hide x-tick labels for the top three plots

    ax.set_ylabel('Relative Brightness')
    ax.grid(True, which='major')

    # Add dashed lines and shaded area for the interval plots
    ax.axvline(x=x_start, color='k', linestyle='--')
    ax.axvline(x=x_end, color='k', linestyle='--')
    ax.axvspan(x_start, x_end, color='red', alpha=0.1)

plt.tight_layout()
plt.show()

# Save
