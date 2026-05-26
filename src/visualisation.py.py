"""
Visualization module for steganography results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
import cv2

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
})

# Color palette for techniques
TECHNIQUE_COLORS = {
    'DWT': '#2196F3',
    'IWT': '#4CAF50',
    'DWT+Palette': '#FF9800',
    'IWT+Palette': '#E91E63',
    'DWT+AdaptPalette': '#9C27B0',
    'IWT+AdaptPalette': '#00BCD4',
}

TECHNIQUE_ORDER = [
    'DWT', 'IWT',
    'DWT+Palette', 'IWT+Palette',
    'DWT+AdaptPalette', 'IWT+AdaptPalette',
]


class SteganoVisualizer:
    """Visualization utilities for steganography results"""
    
    def __init__(self, figures_folder="figures"):
        """
        Initialize visualizer
        
        Args:
            figures_folder: Directory to save figures
        """
        self.figures_folder = figures_folder
        os.makedirs(figures_folder, exist_ok=True)
        self.colors = TECHNIQUE_COLORS
        self.techniques = TECHNIQUE_ORDER
    
    def plot_comparison_bar_chart(self, df, metrics=None, save_name="fig1_comparison.png"):
        """
        Create bar chart comparison of techniques
        
        Args:
            df: DataFrame with results
            metrics: List of metrics to plot
            save_name: Output filename
        """
        if metrics is None:
            metrics = ['PSNR (dB)', 'SSIM', 'MSE', 'NCC', 'UIQI', 'BER']
        
        directions = ['↑Higher', '↑Higher', '↓Lower', '↑Higher', '↑Higher', '↓Lower']
        
        fig, axes = plt.subplots(2, 3, figsize=(13.33, 7.5))
        
        for idx, (metric, direction) in enumerate(zip(metrics, directions)):
            ax = axes[idx // 3, idx % 3]
            
            avgs = [df[df['Technique'] == t][metric].mean() for t in self.techniques]
            stds = [df[df['Technique'] == t][metric].std() for t in self.techniques]
            
            bars = ax.bar(range(len(self.techniques)), avgs, yerr=stds,
                          color=[self.colors[t] for t in self.techniques],
                          edgecolor='black', capsize=4, alpha=0.85)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, avgs)):
                fmt = '.1f' if abs(val) > 10 else '.4f'
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:{fmt}}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')
            
            ax.set_xticks(range(len(self.techniques)))
            ax.set_xticklabels([t.replace('+', '+\n') for t in self.techniques], fontsize=8)
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'{metric} ({direction})', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Technique Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_visual_comparison(self, images, techniques_to_show=None, 
                               n_images=3, save_name="fig2_visual.png"):
        """
        Create visual comparison of cover vs stego images
        
        Args:
            images: Dictionary of images
            techniques_to_show: List of techniques to show
            n_images: Number of images to display
            save_name: Output filename
        """
        if techniques_to_show is None:
            techniques_to_show = ['DWT', 'IWT', 'IWT+Palette', 'IWT+AdaptPalette']
        
        img_names = list(images.keys())[:n_images]
        rows = len(img_names)
        cols = len(techniques_to_show) + 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(13.33, 7.5))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for r, img_name in enumerate(img_names):
            cover = images[img_name]
            
            # Cover image
            axes[r, 0].imshow(cover, cmap='gray')
            if r == 0:
                axes[r, 0].set_title('Cover', fontsize=11, fontweight='bold')
            axes[r, 0].axis('off')
            axes[r, 0].set_ylabel(f'I{r+1}', fontsize=11, fontweight='bold')
            
            # Stego images for each technique
            for c, tech in enumerate(techniques_to_show):
                # This assumes you have a way to generate stego
                # You'll need to pass stego images or a generation function
                stego = cover  # Placeholder - replace with actual stego
                
                axes[r, c + 1].imshow(stego, cmap='gray')
                if r == 0:
                    axes[r, c + 1].set_title(tech, fontsize=10, fontweight='bold')
                axes[r, c + 1].axis('off')
        
        plt.suptitle('Visual Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_histogram_individual(self, images, stegos_dict, 
                                  save_name="fig_hist1_individual.png"):
        """
        Plot individual histograms for cover and stego images
        
        Args:
            images: Dictionary of cover images
            stegos_dict: Nested dict {img_name: {tech: stego_img}}
            save_name: Output filename
        """
        img_names = list(images.keys())[:4]
        rows = len(img_names)
        cols = len(self.techniques) + 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, img_name in enumerate(img_names):
            cover = images[img_name]
            
            # Cover histogram
            axes[row, 0].hist(cover.ravel(), bins=256, range=(0, 256),
                              color='#333333', alpha=0.8, density=True)
            axes[row, 0].set_title(f'Cover\n({img_name})', fontsize=8, fontweight='bold')
            axes[row, 0].set_xlim(0, 256)
            axes[row, 0].tick_params(labelsize=6)
            axes[row, 0].grid(axis='y', alpha=0.2)
            
            if row == len(img_names) - 1:
                axes[row, 0].set_xlabel('Pixel Value', fontsize=7)
            
            # Stego histograms
            for col, tech in enumerate(self.techniques):
                stego = stegos_dict.get(img_name, {}).get(tech, cover)
                color = self.colors[tech]
                
                axes[row, col + 1].hist(stego.ravel(), bins=256, range=(0, 256),
                                         color=color, alpha=0.8, density=True)
                
                # Calculate histogram correlation
                hist_c = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
                hist_s = cv2.calcHist([stego.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
                corr = np.corrcoef(hist_c, hist_s)[0, 1]
                
                axes[row, col + 1].set_title(f'{tech}\nρ={corr:.6f}', 
                                             fontsize=7, fontweight='bold')
                axes[row, col + 1].set_xlim(0, 256)
                axes[row, col + 1].tick_params(labelsize=6)
                axes[row, col + 1].grid(axis='y', alpha=0.2)
                
                if row == len(img_names) - 1:
                    axes[row, col + 1].set_xlabel('Pixel Value', fontsize=7)
        
        plt.suptitle('Histogram Analysis: Cover vs Stego Images\n(ρ = correlation, closer to 1 = better)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_histogram_overlay(self, images, stegos_dict, 
                               save_name="fig_hist2_overlay.png"):
        """
        Plot overlay histograms (cover + stego together)
        
        Args:
            images: Dictionary of cover images
            stegos_dict: Nested dict {img_name: {tech: stego_img}}
            save_name: Output filename
        """
        img_names = list(images.keys())[:4]
        rows = len(img_names)
        cols = len(self.techniques)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, img_name in enumerate(img_names):
            cover = images[img_name]
            
            for col, tech in enumerate(self.techniques):
                stego = stegos_dict.get(img_name, {}).get(tech, cover)
                color = self.colors[tech]
                
                # Overlay both histograms
                axes[row, col].hist(cover.ravel(), bins=256, range=(0, 256),
                                    color='black', alpha=0.4, density=True, label='Cover')
                axes[row, col].hist(stego.ravel(), bins=256, range=(0, 256),
                                    color=color, alpha=0.4, density=True, label='Stego')
                
                # Calculate total difference
                hist_c = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
                hist_s = cv2.calcHist([stego.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
                total_diff = np.sum(np.abs(hist_c - hist_s))
                
                if row == 0:
                    axes[row, col].set_title(tech, fontsize=9, fontweight='bold')
                
                axes[row, col].set_xlim(0, 256)
                axes[row, col].tick_params(labelsize=6)
                axes[row, col].grid(axis='y', alpha=0.2)
                
                axes[row, col].text(0.02, 0.85, f'Δ={total_diff:.0f}',
                                     transform=axes[row, col].transAxes, fontsize=7,
                                     color='red', fontweight='bold',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                if col == 0:
                    axes[row, col].set_ylabel(img_name, fontsize=8)
                
                if row == 0 and col == 0:
                    axes[row, col].legend(fontsize=6, loc='upper right')
        
        plt.suptitle('Overlay Histogram: Cover (Black) vs Stego (Color)\n(Δ = total histogram difference, lower = better)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_histogram_difference(self, images, stegos_dict, 
                                   save_name="fig_hist3_difference.png"):
        """
        Plot histogram differences (stego - cover)
        
        Args:
            images: Dictionary of cover images
            stegos_dict: Nested dict {img_name: {tech: stego_img}}
            save_name: Output filename
        """
        img_names = list(images.keys())[:4]
        rows = len(img_names)
        cols = len(self.techniques)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, img_name in enumerate(img_names):
            cover = images[img_name]
            hist_c = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
            
            for col, tech in enumerate(self.techniques):
                stego = stegos_dict.get(img_name, {}).get(tech, cover)
                hist_s = cv2.calcHist([stego.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
                
                # Difference
                diff = hist_s.astype(float) - hist_c.astype(float)
                
                # Red = positive, Blue = negative
                axes[row, col].bar(range(256), np.maximum(diff, 0), width=1.0,
                                    color='red', alpha=0.6, label='Added')
                axes[row, col].bar(range(256), np.minimum(diff, 0), width=1.0,
                                    color='blue', alpha=0.6, label='Removed')
                axes[row, col].axhline(y=0, color='black', linewidth=0.5)
                
                axes[row, col].set_xlim(0, 256)
                axes[row, col].tick_params(labelsize=6)
                axes[row, col].grid(axis='y', alpha=0.2)
                
                max_change = np.max(np.abs(diff))
                axes[row, col].text(0.02, 0.85, f'Max:{max_change:.0f}',
                                     transform=axes[row, col].transAxes, fontsize=6,
                                     fontweight='bold',
                                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                
                if row == 0:
                    axes[row, col].set_title(tech, fontsize=9, fontweight='bold')
                if col == 0:
                    axes[row, col].set_ylabel(img_name, fontsize=8)
                if row == rows - 1 and col == 0:
                    axes[row, col].legend(fontsize=6, loc='lower right')
        
        plt.suptitle('Histogram Difference: |Stego - Cover|\n(Red = added, Blue = removed, Flatter = better)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_histogram_statistics(self, df_hist, save_name="fig_hist4_statistics.png"):
        """
        Plot histogram statistics comparison
        
        Args:
            df_hist: DataFrame with histogram statistics
            save_name: Output filename
        """
        hist_metrics = [
            ('Correlation', '↑ Higher Better', True),
            ('Chi-Square', '↓ Lower Better', False),
            ('Bhattacharyya', '↓ Lower Better', False),
            ('Total Diff', '↓ Lower Better', False),
            ('KL Divergence', '↓ Lower Better', False),
            ('Intersection', '↑ Higher Better', True),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, (metric, direction, higher_better) in enumerate(hist_metrics):
            ax = axes[idx // 3, idx % 3]
            
            avgs = [df_hist[df_hist['Technique'] == t][metric].mean() for t in self.techniques]
            stds = [df_hist[df_hist['Technique'] == t][metric].std() for t in self.techniques]
            
            best_idx = np.argmax(avgs) if higher_better else np.argmin(avgs)
            
            bars = ax.bar(range(len(self.techniques)), avgs, yerr=stds,
                           color=[self.colors[t] for t in self.techniques],
                           edgecolor='black', capsize=4, alpha=0.85)
            
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            # Value labels
            for i, (bar, val) in enumerate(zip(bars, avgs)):
                if abs(val) < 0.01:
                    fmt = '.6f'
                elif abs(val) < 1:
                    fmt = '.4f'
                else:
                    fmt = '.1f'
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:{fmt}}', ha='center', va='bottom', fontsize=7, fontweight='bold')
            
            ax.set_xticks(range(len(self.techniques)))
            ax.set_xticklabels([t.replace('+', '+\n') for t in self.techniques], fontsize=7)
            ax.set_ylabel(metric, fontsize=10, fontweight='bold')
            ax.set_title(f'{metric}\n({direction})', fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Histogram-Based Security Analysis: Technique Comparison\n(Red border = best, error bars = std dev)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_comprehensive_histogram(self, images, stegos_dict, img_name, 
                                      techniques_to_show=None,
                                      save_name="fig_hist5_comprehensive.png"):
        """
        Create comprehensive histogram analysis for a single image
        
        Args:
            images: Dictionary of cover images
            stegos_dict: Nested dict {img_name: {tech: stego_img}}
            img_name: Name of image to analyze
            techniques_to_show: List of techniques to show
            save_name: Output filename
        """
        if techniques_to_show is None:
            techniques_to_show = ['DWT', 'IWT', 'IWT+Palette', 'IWT+AdaptPalette']
        
        cover = images[img_name]
        n_show = len(techniques_to_show) + 1
        
        fig = plt.figure(figsize=(20, 12))
        
        for col in range(n_show):
            if col == 0:
                img = cover
                title = f'Cover Image\n({img_name})'
                color = '#333333'
            else:
                tech = techniques_to_show[col - 1]
                img = stegos_dict.get(img_name, {}).get(tech, cover)
                title = tech
                color = self.colors[tech]
            
            # Row 1: Image
            ax_img = fig.add_subplot(3, n_show, col + 1)
            ax_img.imshow(img, cmap='gray')
            ax_img.set_title(title, fontsize=11, fontweight='bold')
            ax_img.axis('off')
            
            # Row 2: Individual histogram
            ax_hist = fig.add_subplot(3, n_show, n_show + col + 1)
            ax_hist.hist(img.ravel(), bins=256, range=(0, 256),
                          color=color, alpha=0.8, density=True)
            ax_hist.set_xlim(0, 256)
            ax_hist.set_xlabel('Pixel Value', fontsize=9)
            if col == 0:
                ax_hist.set_ylabel('Density', fontsize=9)
            ax_hist.tick_params(labelsize=7)
            ax_hist.grid(axis='y', alpha=0.2)
            
            # Row 3: Overlay with cover
            ax_over = fig.add_subplot(3, n_show, 2 * n_show + col + 1)
            
            if col == 0:
                ax_over.hist(cover.ravel(), bins=256, range=(0, 256),
                              color='black', alpha=0.7, density=True)
                ax_over.set_title('Reference', fontsize=9)
            else:
                ax_over.hist(cover.ravel(), bins=256, range=(0, 256),
                              color='black', alpha=0.4, density=True, label='Cover')
                ax_over.hist(img.ravel(), bins=256, range=(0, 256),
                              color=color, alpha=0.4, density=True, label='Stego')
                
                # Calculate and show correlation
                h_c = cv2.calcHist([cover], [0], None, [256], [0, 256]).flatten()
                h_s = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
                corr = np.corrcoef(h_c, h_s)[0, 1]
                
                text_color = 'green' if corr > 0.9999 else 'orange' if corr > 0.999 else 'red'
                ax_over.set_title(f'ρ = {corr:.6f}', fontsize=10, fontweight='bold', color=text_color)
                ax_over.legend(fontsize=7)
            
            ax_over.set_xlim(0, 256)
            ax_over.set_xlabel('Pixel Value', fontsize=9)
            if col == 0:
                ax_over.set_ylabel('Overlay', fontsize=9)
            ax_over.tick_params(labelsize=7)
            ax_over.grid(axis='y', alpha=0.2)
        
        plt.suptitle(f'Comprehensive Histogram Analysis — {img_name}\n'
                     f'Row 1: Images | Row 2: Histograms | Row 3: Overlay with Cover',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_robustness_heatmap(self, df_rob, save_name="fig_robustness_heatmap.png"):
        """
        Create heatmap of robustness results
        
        Args:
            df_rob: DataFrame with robustness results
            save_name: Output filename
        """
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create pivot table
        pivot = df_rob.pivot_table(
            index='Technique', 
            columns='Attack', 
            values='BER After Attack',
            aggfunc='mean'
        )
        
        # Sort techniques
        pivot = pivot.reindex(self.techniques)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Custom colormap (green = low BER, red = high BER)
        colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list('ber_cmap', colors, N=256)
        
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([a.replace(' ', '\n') for a in pivot.columns], fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=10)
        
        # Rotate x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.4f}',
                               ha='center', va='center', fontsize=7,
                               color='white' if pivot.values[i, j] > 0.25 else 'black')
        
        ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Technique', fontsize=12, fontweight='bold')
        ax.set_title('Robustness Analysis: BER After Attacks\n(Lower is better → green)',
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")
    
    def plot_capacity_vs_quality(self, df, save_name="fig_capacity_quality.png"):
        """
        Plot capacity vs quality trade-off
        
        Args:
            df: DataFrame with results
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for tech in self.techniques:
            tech_df = df[df['Technique'] == tech]
            x = tech_df['Capacity (bpp)'].values
            y = tech_df['PSNR (dB)'].values
            
            ax.scatter(x, y, s=100, c=[self.colors[tech]], 
                      label=tech, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Capacity (bits per pixel)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('Capacity vs Quality Trade-off\n(Higher PSNR & Higher Capacity = Better)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_folder, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {save_path}")