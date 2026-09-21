# ============================================================
# F1 VEHICLE PERFORMANCE MODEL
# ============================================================
#
# Longitudinal vehicle dynamics model
#
# Includes:
#   - Engine power
#   - Tire traction limit
#   - Aerodynamic drag
#   - Aerodynamic downforce
#   - Rolling resistance
#   - Speed-dependent tire grip
#   - Acceleration
#   - Top speed estimation
#   - Performance graphs
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VEHICLE PARAMETERS
# ============================================================

# Vehicle mass [kg]
mass = 800.0

# Maximum engine power [W]
engine_power = 750_000.0

# Drivetrain efficiency [-]
# 1.0 = 100% efficient
drivetrain_efficiency = 0.95

# Effective power at the driven wheels [W]
wheel_power = engine_power * drivetrain_efficiency


# ============================================================
# AERODYNAMIC PARAMETERS
# ============================================================

# Air density [kg/m^3]
air_density = 1.225

# Frontal area [m^2]
frontal_area = 1.5

# Drag coefficient [-]
drag_coefficient = 0.90

# Downforce coefficient [-]
downforce_coefficient = 2.0


# ============================================================
# TIRE / ROAD PARAMETERS
# ============================================================

# Tire-road friction coefficient [-]
tire_grip = 1.50

# Rolling resistance coefficient [-]
rolling_resistance = 0.015

# Gravitational acceleration [m/s^2]
g = 9.81


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

# Simulation speed range [km/h]
max_simulation_speed_kmh = 400.0

# Speed step [km/h]
speed_step_kmh = 1.0


# ============================================================
# SPEED ARRAY
# ============================================================

speeds_kmh = np.arange(
    1.0,
    max_simulation_speed_kmh + speed_step_kmh,
    speed_step_kmh
)

# Convert km/h -> m/s
speeds_ms = speeds_kmh / 3.6


# ============================================================
# AERODYNAMIC FORCES
# ============================================================

# Aerodynamic drag:
#
# F_drag = 0.5 * rho * Cd * A * v^2

drag_forces = (
    0.5
    * air_density
    * drag_coefficient
    * frontal_area
    * speeds_ms**2
)


# Aerodynamic downforce:
#
# F_downforce = 0.5 * rho * Cl * A * v^2

downforce_forces = (
    0.5
    * air_density
    * downforce_coefficient
    * frontal_area
    * speeds_ms**2
)


# ============================================================
# NORMAL FORCE
# ============================================================

# Total vertical load on the tires:
#
# N = m*g + downforce

normal_forces = (
    mass * g
    + downforce_forces
)


# ============================================================
# TIRE TRACTION LIMIT
# ============================================================

# Maximum longitudinal tire force:
#
# F_tire = mu * N

tire_force_limits = (
    tire_grip
    * normal_forces
)


# ============================================================
# ENGINE / POWER FORCE
# ============================================================

# Power available at the wheels:
#
# F = P / v

power_forces = (
    wheel_power
    / speeds_ms
)


# ============================================================
# ACTUAL TRACTIVE FORCE
# ============================================================

# The vehicle can only use whichever force is lower:
#
#   tire limit
#       OR
#   power limit

tractive_forces = np.minimum(
    tire_force_limits,
    power_forces
)


# ============================================================
# ROLLING RESISTANCE
# ============================================================

# Rolling resistance:
#
# F_roll = Crr * N

rolling_forces = (
    rolling_resistance
    * normal_forces
)


# ============================================================
# NET FORCE
# ============================================================

net_forces = (
    tractive_forces
    - drag_forces
    - rolling_forces
)


# ============================================================
# ACCELERATION
# ============================================================

accelerations = (
    net_forces
    / mass
)


# ============================================================
# TRANSITION FROM TIRE-LIMITED TO POWER-LIMITED
# ============================================================

# Find where power force becomes lower than tire force.

power_limited_indices = np.where(
    power_forces < tire_force_limits
)[0]

if len(power_limited_indices) > 0:

    transition_index = power_limited_indices[0]

    transition_speed_kmh = (
        speeds_kmh[transition_index]
    )

else:

    transition_speed_kmh = None


# ============================================================
# TOP SPEED
# ============================================================

# Theoretical top speed occurs when:
#
# net force = 0
#
# i.e.
#
# tractive force = drag + rolling resistance

top_speed_indices = np.where(
    net_forces <= 0
)[0]

if len(top_speed_indices) > 0:

    top_speed_index = top_speed_indices[0]

    top_speed_kmh = (
        speeds_kmh[top_speed_index]
    )

else:

    top_speed_kmh = None


# ============================================================
# MAXIMUM ACCELERATION
# ============================================================

max_acceleration_index = np.argmax(
    accelerations
)

max_acceleration = (
    accelerations[max_acceleration_index]
)

max_acceleration_speed = (
    speeds_kmh[max_acceleration_index]
)


# ============================================================
# 0-100 KM/H ESTIMATION
# ============================================================

# Numerical integration:
#
# dt = dv / a

time = 0.0

zero_to_100_time = None

for i in range(1, len(speeds_ms)):

    acceleration = accelerations[i]

    if acceleration <= 0:
        break

    delta_v = (
        speeds_ms[i]
        - speeds_ms[i - 1]
    )

    delta_t = (
        delta_v
        / acceleration
    )

    time += delta_t

    if speeds_kmh[i] >= 100:

        zero_to_100_time = time
        break


# ============================================================
# 0-200 KM/H ESTIMATION
# ============================================================

time = 0.0

zero_to_200_time = None

for i in range(1, len(speeds_ms)):

    acceleration = accelerations[i]

    if acceleration <= 0:
        break

    delta_v = (
        speeds_ms[i]
        - speeds_ms[i - 1]
    )

    delta_t = (
        delta_v
        / acceleration
    )

    time += delta_t

    if speeds_kmh[i] >= 200:

        zero_to_200_time = time
        break


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print()
print("============================================================")
print("                 F1 VEHICLE PERFORMANCE MODEL")
print("============================================================")
print()

print("VEHICLE")
print("------------------------------------------------------------")
print(f"Mass:                    {mass:.0f} kg")
print(f"Engine power:            {engine_power / 1000:.0f} kW")
print(f"Wheel power:             {wheel_power / 1000:.0f} kW")
print()

print("AERODYNAMICS")
print("------------------------------------------------------------")
print(f"Drag coefficient:        {drag_coefficient:.2f}")
print(f"Downforce coefficient:   {downforce_coefficient:.2f}")
print(f"Frontal area:             {frontal_area:.2f} m²")
print()

print("TIRE / ROAD")
print("------------------------------------------------------------")
print(f"Tire grip coefficient:    {tire_grip:.2f}")
print(f"Rolling resistance:       {rolling_resistance:.3f}")
print()

print("PERFORMANCE")
print("------------------------------------------------------------")

if transition_speed_kmh is not None:
    print(
        f"Traction -> power transition: "
        f"{transition_speed_kmh:.1f} km/h"
    )
else:
    print(
        "Traction -> power transition: "
        "not reached"
    )

if top_speed_kmh is not None:
    print(
        f"Theoretical top speed:        "
        f"{top_speed_kmh:.1f} km/h"
    )
else:
    print(
        "Theoretical top speed:        "
        "not reached"
    )

print(
    f"Maximum acceleration:          "
    f"{max_acceleration:.2f} m/s²"
)

print(
    f"Maximum acceleration speed:    "
    f"{max_acceleration_speed:.1f} km/h"
)

if zero_to_100_time is not None:
    print(
        f"Estimated 0-100 km/h:          "
        f"{zero_to_100_time:.2f} s"
    )
else:
    print(
        "Estimated 0-100 km/h:          "
        "not available"
    )

if zero_to_200_time is not None:
    print(
        f"Estimated 0-200 km/h:          "
        f"{zero_to_200_time:.2f} s"
    )
else:
    print(
        "Estimated 0-200 km/h:          "
        "not available"
    )

print()
print("============================================================")


# ============================================================
# GRAPH 1 - ACCELERATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    speeds_kmh,
    accelerations,
    label="Acceleration"
)

if transition_speed_kmh is not None:

    plt.axvline(
        transition_speed_kmh,
        linestyle="--",
        label=(
            f"Traction / Power transition "
            f"({transition_speed_kmh:.1f} km/h)"
        )
    )

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Speed [km/h]")
plt.ylabel("Acceleration [m/s²]")
plt.title("F1 Vehicle Acceleration")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ============================================================
# GRAPH 2 - FORCES
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    speeds_kmh,
    tractive_forces,
    label="Tractive force"
)

plt.plot(
    speeds_kmh,
    drag_forces,
    label="Aerodynamic drag"
)

plt.plot(
    speeds_kmh,
    rolling_forces,
    label="Rolling resistance"
)

plt.plot(
    speeds_kmh,
    tire_force_limits,
    linestyle="--",
    label="Tire force limit"
)

plt.xlabel("Speed [km/h]")
plt.ylabel("Force [N]")
plt.title("F1 Vehicle Forces")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ============================================================
# GRAPH 3 - AERODYNAMIC DOWNFORCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    speeds_kmh,
    downforce_forces,
    label="Downforce"
)

plt.xlabel("Speed [km/h]")
plt.ylabel("Downforce [N]")
plt.title("Aerodynamic Downforce")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ============================================================
# GRAPH 4 - TRACTIVE FORCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    speeds_kmh,
    tractive_forces,
    label="Actual tractive force"
)

plt.plot(
    speeds_kmh,
    power_forces,
    linestyle="--",
    label="Power-limited force"
)

plt.plot(
    speeds_kmh,
    tire_force_limits,
    linestyle=":",
    label="Tire-limited force"
)

plt.xlabel("Speed [km/h]")
plt.ylabel("Force [N]")
plt.title("Tractive Force Limits")

plt.grid(True)
plt.legend()

plt.tight_layout()


# ============================================================
# SHOW ALL GRAPHS
# ============================================================

plt.show()
