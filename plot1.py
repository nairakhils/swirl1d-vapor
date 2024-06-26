import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('swirl1d.csv', names=['x', 'density', 'velocity', 'pressure'])

# Create the plots
plt.figure(figsize=(10, 10))

# Density
plt.subplot(3, 1, 1)
plt.plot(data['x'], data['density'])
plt.title('Density')
plt.xlabel('x')
plt.ylabel('Density')

# Velocity
plt.subplot(3, 1, 2)
plt.plot(data['x'], data['velocity'])
plt.title('Velocity')
plt.xlabel('x')
plt.ylabel('Velocity')

# Pressure
plt.subplot(3, 1, 3)
plt.plot(data['x'], data['pressure'])
plt.title('Pressure')
plt.xlabel('x')
plt.ylabel('Pressure')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
