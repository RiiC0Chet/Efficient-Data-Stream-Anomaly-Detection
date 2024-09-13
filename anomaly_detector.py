import numpy as np
import matplotlib.pyplot as plt
from collections import deque

"""
Simulates a data stream with a variable sinusoidal pattern, random noise, and occasional anomalies.

:param length: Number of data points in the stream
:param anomaly_prob: Probability of generating an anomaly
:param noise_level: Standard deviation of the added Gaussian noise
:return: Generator that yields simulated data points
"""
def data_stream_simulation(length=1000, anomaly_prob=0.01, noise_level=0.1):

    # Validate input parameters
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if not (0 <= anomaly_prob <= 1):
        raise ValueError("anomaly_prob must be between 0 and 1")
    if not isinstance(noise_level, (int, float)) or noise_level < 0:
        raise ValueError("noise_level must be a non-negative number")

    # First, we create a static data stream with np.arange, then convert it into a sinusoidal function 
    # with various wave frequencies, simulating stock market movements. After that, noise and  
    # anomalies in the form of spikes will be added to the graph.
    
    t = np.arange(0, length, 0.1)  # Time variable for the main wave without addons
    
    # Set the parameters for wave frequency, wave center, and wave amplitude
    
    frequency = 1  # Regular frequency of the main wave
    center_wave = np.sin(0.05 * t) * 2  # Low-frequency wave for the center, amplitude of 2
    amplitude_wave = np.abs(np.sin(0.1 * t) + np.random.normal(0, 0.1, len(t))) * 3  # Amplitude between 0 and 3
    
    # In the for loop, we will create a point of the sinusoidal wave with its noise and anomaly based on the 
    # anomaly probability for each call to the generator
    
    for i in range(len(t)):
        center = center_wave[i]  # Center of the current sinusoidal wave
        amplitude = amplitude_wave[i]  # Amplitude of the current sinusoidal wave
        seasonal_pattern = center + amplitude * np.sin(frequency * t[i])
        noise = np.random.normal(0, noise_level)
        
        anomaly = 0
        if np.random.rand() < anomaly_prob:
            anomaly = np.random.uniform(-5, 5)  # Large anomaly spike
        
        yield seasonal_pattern + noise + anomaly

"""
Detects anomalies in a data stream using Z-score over a sliding window.

:param data_stream: Generator that yields data points
:param window_size: Number of data points in the sliding window
:param threshold: Z-score threshold for flagging anomalies
:return: Generator yielding (data_point, is_anomaly) tuples
"""
def z_score_anomaly_detection(data_stream, window_size=50, threshold=3):

    # Validate input parameters
    if not hasattr(data_stream, '__iter__'):
        raise ValueError("data_stream must be an iterable or generator")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError("threshold must be a positive number")

    # Declare local variables for the function, such as the sliding window, mean, and variance
    
    window = deque(maxlen=window_size)
    mean = 0
    variance = 0
    
    for data_point in data_stream:
        
        # Validate that the data point is a number
        
        if not isinstance(data_point, (int, float)):
            raise ValueError("data_point must be a number")

        # When the window is full, calculate the mean and variance for each iteration of the generator
        
        if len(window) == window_size:
            mean = np.mean(window)
            variance = np.var(window)
        
        # Add the new data value to the end of the window, implicitly removing the first one
        
        window.append(data_point)
        
        # If the window is not full, we cannot calculate the mean and variance
        if len(window) < window_size:
            yield data_point, False
        else: # Once the window is full, we can start detecting anomalies using the z-score function
            std_dev = np.sqrt(variance)
            z_score = (data_point - mean) / std_dev if std_dev != 0 else 0
            is_anomaly = bool(abs(z_score) > threshold)  # Ensure is_anomaly is a boolean
            yield data_point, is_anomaly

"""
Visualizes the data stream in real-time, updating the plot dynamically as data is processed.

:param data_stream: Generator that yields data points
:param window_size: Number of data points in the sliding window for anomaly detection
:param threshold: Z-score threshold for flagging anomalies
"""
def real_time_visualization(data_stream, window_size=50, threshold=3):

    # Validate input parameters
    if not hasattr(data_stream, '__iter__'):
        raise ValueError("data_stream must be an iterable or generator")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError("threshold must be a positive number")

    # Set up the plot
    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1000)  # Initial x-axis limits
    ax.set_ylim(-10, 10)  # Initial y-axis limits
    line, = ax.plot([], [], label="Data Stream", lw=2)
    anomaly_scatter = ax.scatter([], [], color='red', label="Anomalies", zorder=5)
    
    # Initialize empty lists to store data and anomalies
    data = []
    anomalies_x = []
    anomalies_y = []
    
    anomaly_detector = z_score_anomaly_detection(data_stream, window_size, threshold)

    # Variable to control the loop when the window is closed
    running = [True]

    # Event handler for window close
    def on_close(event):
        running[0] = False
        plt.close(fig)

    # Connect the window close event
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Process each data point from the stream
    for i, (data_point, is_anomaly) in enumerate(anomaly_detector):
        if not running[0]:
            break  # Exit the loop if the window was closed
        
        data.append(data_point)
        
        # Update the line plot with new data
        line.set_data(np.arange(len(data)), data)
        
        if is_anomaly:
            # If an anomaly, update the scatter plot
            anomalies_x.append(i)
            anomalies_y.append(data_point)
        
        # Update scatter plot data
        anomaly_scatter.set_offsets(np.column_stack([anomalies_x, anomalies_y]))
        
        # Rescale the x-axis if necessary
        if i >= 1000:
            ax.set_xlim(i - 1000, i)
        
        # Adjust the plot limits if data goes beyond current y-axis limits
        if data_point > ax.get_ylim()[1]:
            ax.set_ylim(ax.get_ylim()[0], data_point + 5)
        elif data_point < ax.get_ylim()[0]:
            ax.set_ylim(data_point - 5, ax.get_ylim()[1])
        
        plt.legend()
        plt.pause(0.01)  # Short pause to update the plot in real-time
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Main entry point for testing the real-time visualization
if __name__ == "__main__":
    try:
        # Simulate a non-linear sinusoidal data stream
        stream = data_stream_simulation(length=1000, anomaly_prob=0.02, noise_level=0.2)
        
        # Visualize the data and anomalies in real-time
        real_time_visualization(stream, window_size=50, threshold=2)
    except Exception as e:
        print(f"An error occurred: {e}")