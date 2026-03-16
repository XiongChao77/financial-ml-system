import numpy as np
import matplotlib.pyplot as plt

def plot_enhanced_comparison(scale=1.0):
    # Generate input range
    x = np.linspace(-5 * scale, 5 * scale, 1000)
    
    # Compute function values
    y_tanh = np.tanh(x / scale)
    y_log = np.sign(x) * np.log1p(np.abs(x / scale))
    
    plt.figure(figsize=(14, 8), dpi=100)
    
    # 1. Plot curves
    plt.plot(x, y_tanh, label=r'$\tanh(x/S)$', color='#E74C3C', lw=3)
    plt.plot(x, y_log, label=r'$\text{sgn}(x)\ln(1+|x/S|)$', color='#3498DB', lw=3)
    plt.plot(x, x/scale, '--', label='Linear Reference ($y=x/S$)', color='#95A5A6', alpha=0.4)

    # 2. Key points (x/S = 1, 2, 3)
    key_points = [1.0, 2.0, 3.0]
    colors = ['#27AE60', '#F39C12', '#8E44AD']
    
    for kp in key_points:
        curr_x = kp * scale
        v_tanh = np.tanh(kp)
        v_log = np.log1p(kp)
        
        # Vertical guide line
        plt.axvline(curr_x, color='gray', linestyle=':', alpha=0.3)
        
        # Annotate tanh key point
        plt.scatter(curr_x, v_tanh, color='#E74C3C', zorder=5)
        plt.text(curr_x + 0.1, v_tanh - 0.05, f'tanh({kp})≈{v_tanh:.3f}', color='#C0392B', fontweight='bold')
        
        # Annotate log1p key point
        plt.scatter(curr_x, v_log, color='#3498DB', zorder=5)
        plt.text(curr_x + 0.1, v_log + 0.05, f'ln(1+{kp})≈{v_log:.3f}', color='#2980B9', fontweight='bold')

    # 3. Shade saturation zones
    # Linear-ish zone: |x| < S
    plt.axvspan(-scale, scale, color='#2ECC71', alpha=0.1, label='Quasi-Linear Zone')
    # Tanh saturation: |x| > 2.5S
    plt.axvspan(2.5 * scale, 5 * scale, color='#E74C3C', alpha=0.05)
    plt.axvspan(-5 * scale, -2.5 * scale, color='#E74C3C', alpha=0.05, label='Tanh Saturation')

    # 4. Chart styling
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.ylim(-2.0, 2.0)
    plt.xlim(-5 * scale, 5 * scale)
    
    plt.title(f'Quantitative Perspective: Tanh vs Log1p (Scale S={scale})', fontsize=16, pad=20)
    plt.xlabel('Input Standardized Signal ($x$)', fontsize=12)
    plt.ylabel('Normalized Output ($y$)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Add explanatory text
    info_text = (
        "Key Insights:\n"
        "1. Linear Zone: x < S, both are reliable.\n"
        "2. Soft Saturation: x = 2S, Tanh hits 0.96 (Heavy Squashing).\n"
        "3. Diversity: At x = 3S, Log1p (1.39) still grows while Tanh (0.99) is flat."
    )
    plt.text(-4.8 * scale, -1.8, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

plot_enhanced_comparison(scale=1.0)