import json
import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data, **kwargs):
    x = [d['length'] for d in data]
    y = [d['gputime'] for d in data]
    plt.plot(x, y, linewidth=1.5, **kwargs)

def plot_qua(data, x_fit=None, **kwargs):
    x = [d['length'] for d in data]
    y = [d['gputime'] for d in data]
    coefficients = np.polyfit(x, y, 2)
    a, b, c = coefficients
    print(f"y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}")
    if x_fit is None:
        x_fit = np.linspace(1000, 256000, 129)
    y_fit = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, y_fit, **kwargs)

def plot_lin(data, x_fit=None, **kwargs):
    x = [d['length'] for d in data]
    y = [d['gputime'] for d in data]
    coefficients = np.polyfit(x, y, 1)
    a, b = coefficients
    print(f"y = {a:.2f}x + {b:.2f}")
    if x_fit is None:
        x_fit = np.linspace(1000, 256000, 129)
    y_fit = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, y_fit, **kwargs)

with open('outputs/Efficiency-Quantized.jsonl', 'r') as f:
    baseline = list(map(json.loads, f.readlines()))
# with open('outputs/Efficiency-Quantized.jsonl', 'r') as f:
#     quantized = list(map(json.loads, f.readlines()))
with open('outputs/Efficiency-TTTTuned.jsonl', 'r') as f:
    ttt = list(map(json.loads, f.readlines()))
# with open('outputs/Efficiency-SFT+TTT.jsonl', 'r') as f:
#     sftttt = list(map(json.loads, f.readlines()))

plt.xlabel('Length/token')
plt.ylabel('GPUTime/s')
plot_curve(baseline, label='ICL', color='#00FF00', marker='o')
plot_qua(baseline, x_fit=[90000], color='red', marker='X', markersize=12)
plot_qua(baseline, color='#006400', linestyle='--')
# plot_curve(quantized, label='Quantized', color='#006400', linestyle='--', marker='o')
plot_curve(ttt, label='LIFT', color='#0000FF', marker='o')
plot_lin(ttt, color='#00008B', linestyle='--')
# plot_curve(sftttt, label='SFT+TTT', color='#00008B', linestyle='--', marker='o')
plt.legend()
plt.savefig('outputs/efficiency.svg')
