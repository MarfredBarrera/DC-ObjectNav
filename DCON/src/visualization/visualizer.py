import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import cv2
import imageio.v2 as imageio

class Visualizer:
    """
    Unified Visualization class for DC-ObjectNav.
    Handles plotting of BEV uncertainty, similarity, occupancy maps, and planned paths.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        # Default colormaps
        self.occ_cmap = ListedColormap(['#808080', '#FFFFFF', '#000000']) # 0=Unseen (Gray), 1=Free (White), 2=Occupied (Black)
        self.sim_cmap = 'jet'
        self.unc_cmap = 'magma'
        
    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def normalize_sim(self, m):
        """Normalize similarity map to [0, 1] range, ignoring zeros."""
        if m is None: return None
        m = self._to_numpy(m)
        non_zero_mask = m > 0
        if not np.any(non_zero_mask): return m
        m_min, m_max = m[non_zero_mask].min(), m[non_zero_mask].max()
        m_norm = np.zeros_like(m)
        if m_max > m_min:
            m_norm[non_zero_mask] = (m[non_zero_mask] - m_min) / (m_max - m_min)
        else:
            m_norm[non_zero_mask] = 1.0
        return m_norm

    def apply_temperature(self, sim_map, temperature=0.2):
        """Apply Softmax temperature scaling and re-normalize."""
        if temperature == 1.0 or sim_map is None:
            return sim_map
        sim_map = self._to_numpy(sim_map)
        max_val = np.max(sim_map)
        exp_map = np.exp((sim_map - max_val) / temperature)
        softmax_map = exp_map / np.sum(exp_map)
        if np.max(softmax_map) > 0:
            return softmax_map / np.max(softmax_map)
        return softmax_map

    def get_z_score(self, m, mask=None, ignore_percentile=0):
        """Compute Z-score of a map, optionally with a mask and percentile filter."""
        m = self._to_numpy(m)
        if mask is not None:
            mask = self._to_numpy(mask)
            m_vals = m[mask]
        else:
            m_vals = m.flatten()

        if ignore_percentile > 0 and len(m_vals) > 0:
            thresh = np.percentile(m_vals, ignore_percentile)
            m_vals = m_vals[m_vals >= thresh]

        if len(m_vals) == 0:
            return np.zeros_like(m)

        mean, std = np.mean(m_vals), np.std(m_vals)
        if std < 1e-8: return np.zeros_like(m)
        return (m - mean) / std

    def plot_uncertainty(self, epi_map, ale_map, extent, step=None, save_path=None):
        """Visualize Epistemic and Aleatoric uncertainty maps."""
        epi_map = self._to_numpy(epi_map)
        ale_map = self._to_numpy(ale_map)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, (m, title, label) in enumerate([
            (epi_map, r"Epistemic: $\mathbb{V}[\mu_\theta]$", "Epi"),
            (ale_map, r"Aleatoric: $\mathbb{E}[\sigma^2_\theta]$", "Ale")
        ]):
            im = axes[i].imshow(m, cmap=self.unc_cmap, origin='lower', aspect='equal', extent=extent)
            axes[i].set_title(title, fontsize=12)
            axes[i].set_xlabel('X (m)')
            axes[i].set_ylabel('Z (m)')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            stats = f'Min: {m.min():.6f}\nMax: {m.max():.6f}\nMean: {m.mean():.6f}'
            axes[i].text(0.02, 0.98, stats, transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
            axes[i].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved uncertainty plot to {save_path}")
        plt.show()
        return fig

    def plot_occupancy(self, occ_map, extent, step=None, save_path=None):
        """Visualize the occupancy map."""
        occ_map = self._to_numpy(occ_map)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(occ_map, cmap=self.occ_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=2)
        ax.set_title(f"Occupancy Map" + (f" (Step {step})" if step else ""), fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.33, 1.0, 1.66])
        cbar.set_ticklabels(['Unseen', 'Free', 'Occupied'])
        
        # Stats
        total = occ_map.size
        explored = (np.sum(occ_map >= 1) / total) * 100
        occupied = (np.sum(occ_map == 2) / total) * 100
        stats = f'Explored: {explored:.2f}%\nOccupied: {occupied:.2f}%'
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig

    def plot_similarity(self, sim_map, extent, step=None, save_path=None, temperature=None):
        """Visualize the similarity map."""
        sim_map = self.normalize_sim(sim_map)
        if temperature:
            sim_map = self.apply_temperature(sim_map, temperature)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_map, cmap=self.sim_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=1)
        ax.set_title(f"Similarity Map" + (f" (Step {step})" if step else ""), fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Similarity Score')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig

    def render_combined_grid(self, maps_dict, extent, step=None, save_path=None):
        """
        Renders a 2x3 grid of maps:
        [RGB, ViewSim, CombinedZ]
        [Uncertainty, Occupancy, BEVSim]
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axs = axes.flatten()
        
        # 0. RGB
        if 'rgb' in maps_dict and maps_dict['rgb'] is not None:
            axs[0].imshow(maps_dict['rgb'])
            axs[0].set_title(f"Agent View (Step {step})" if step else "Agent View")
            axs[0].axis('off')
        
        # 1. View Similarity (2D)
        if 'sim2d' in maps_dict and maps_dict['sim2d'] is not None:
            s2d = self.normalize_sim(maps_dict['sim2d'])
            im1 = axs[1].imshow(s2d, cmap=self.sim_cmap, vmin=0, vmax=1)
            axs[1].set_title("View Similarity")
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02, shrink=0.7)
            axs[1].axis('off')

        # 2. Combined Z-Score
        if 'combined_z' in maps_dict and maps_dict['combined_z'] is not None:
            cz = maps_dict['combined_z']
            im2 = axs[2].imshow(cz, cmap=self.sim_cmap, origin='lower', extent=extent, vmin=-3, vmax=3)
            axs[2].set_title("Combined Z-Score (Sim+Unc)")
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02, shrink=0.7)
            axs[2].grid(True, alpha=0.3)

        # 3. Uncertainty
        if 'epi' in maps_dict and maps_dict['epi'] is not None:
            im3 = axs[3].imshow(maps_dict['epi'], cmap=self.unc_cmap, origin='lower', extent=extent)
            axs[3].set_title("Epistemic Uncertainty")
            plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.02, shrink=0.7)
            axs[3].grid(True, alpha=0.3)

        # 4. Occupancy
        if 'occ' in maps_dict and maps_dict['occ'] is not None:
            im4 = axs[4].imshow(maps_dict['occ'], cmap=self.occ_cmap, origin='lower', extent=extent, vmin=0, vmax=2)
            axs[4].set_title("Occupancy Map")
            cbar = plt.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.02, shrink=0.7)
            cbar.set_ticks([0.33, 1.0, 1.66]); cbar.set_ticklabels(['U', 'F', 'O'])
            axs[4].grid(True, alpha=0.3)

        # 5. BEV Similarity
        if 'sim' in maps_dict and maps_dict['sim'] is not None:
            s_bev = self.normalize_sim(maps_dict['sim'])
            im5 = axs[5].imshow(s_bev, cmap=self.sim_cmap, origin='lower', extent=extent, vmin=0, vmax=1)
            axs[5].set_title("BEV Similarity Map")
            plt.colorbar(im5, ax=axs[5], fraction=0.046, pad=0.02, shrink=0.7)
            axs[5].grid(True, alpha=0.3)

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig

    def plot_planner_paths(self, occ_map, sim_map, epi_map, extent, scores, best_idx, save_path=None, grid_to_world_fn=None):
        """
        Standardized planner visualization with 3 panels:
        1. Paths over Occupancy
        2. Best path over Similarity
        3. FOV shadow over Uncertainty
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Occupancy with all paths
        axes[0].imshow(occ_map, cmap=self.occ_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=2)
        axes[0].set_title("A* Routes over Occupancy")
        for s in scores:
            route = s['traj']
            if route and grid_to_world_fn:
                coords = [grid_to_world_fn(pt[0], pt[1]) for pt in route]
                x_vals, z_vals = [c[0] for c in coords], [c[1] for c in coords]
                alpha = 1.0 if s['idx'] == best_idx else 0.3
                color = 'red' if s['idx'] == best_idx else 'blue'
                axes[0].plot(x_vals, z_vals, color=color, alpha=alpha, linewidth=2, label='Optimized' if s['idx'] == best_idx else None)
                
            # Plot reference trajectory if it exists (A* nominal path)
            ref_route = s.get('ref_traj')
            if ref_route and grid_to_world_fn:
                ref_coords = [grid_to_world_fn(pt[0], pt[1]) for pt in ref_route]
                rx_vals, rz_vals = [c[0] for c in ref_coords], [c[1] for c in ref_coords]
                axes[0].plot(rx_vals, rz_vals, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Reference' if s['idx'] == best_idx else None)

            if route and grid_to_world_fn:
                axes[0].scatter(x_vals[-1], z_vals[-1], marker='x', color='green', s=100)
        
        axes[0].legend(loc='upper right', fontsize=8)

        # 2. Similarity + Best path
        # sim_vis = self.normalize_sim(sim_map)
        sim_mask = (occ_map == 2)
        sim_vis = self.get_z_score(sim_map, mask=sim_mask, ignore_percentile=75)
        im_sim = axes[1].imshow(sim_vis, cmap=self.sim_cmap, origin='lower', aspect='equal', extent=extent, vmin = -3, vmax = 3)
        axes[1].set_title("Best Trajectory over Similarity")
        if best_idx is not None and grid_to_world_fn:
            best_score = scores[best_idx]
            best_traj = best_score['traj']
            best_coords = [grid_to_world_fn(pt[0], pt[1]) for pt in best_traj]
            
            # Plot Reference if it exists
            ref_traj = best_score.get('ref_traj')
            if ref_traj:
                ref_coords = [grid_to_world_fn(pt[0], pt[1]) for pt in ref_traj]
                axes[1].plot([c[0] for c in ref_coords], [c[1] for c in ref_coords], color='orange', linestyle='--', linewidth=2, alpha=0.6)
            
            axes[1].plot([c[0] for c in best_coords], [c[1] for c in best_coords], color='white', linewidth=3, alpha=0.8)
            axes[1].scatter(best_coords[-1][0], best_coords[-1][1], marker='*', color='white', s=200)
            plt.colorbar(im_sim, ax=axes[1], fraction=0.046, pad=0.02, shrink=0.7)

        # 3. Uncertainty + FOV mask
        axes[2].imshow(epi_map, cmap=self.unc_cmap, origin='lower', aspect='equal', extent=extent)
        if best_idx is not None:
            best_score = scores[best_idx]
            axes[2].set_title(f"Info Gain Sweep (IG={best_score.get('ig', 0):.2f})")
            if grid_to_world_fn:
                best_coords = [grid_to_world_fn(pt[0], pt[1]) for pt in best_score['traj']]
                axes[2].plot([c[0] for c in best_coords], [c[1] for c in best_coords], color='cyan', linewidth=3)
            
            seen_mask = best_score.get('seen_mask')
            if seen_mask is not None:
                mask_rgba = np.zeros((*seen_mask.shape, 4))
                mask_rgba[seen_mask] = [0, 1, 0, 0.3] # Green transparent
                axes[2].imshow(mask_rgba, origin='lower', aspect='equal', extent=extent)

        for ax in axes:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig

    def fig_to_numpy(self, fig):
        """Convert a Matplotlib figure to an RGB numpy array."""
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img

    def create_video(self, frames, save_path, fps=2):
        """Save a list of RGB frames as an MP4 or GIF."""
        if not frames:
            print("No frames provided for video creation.")
            return
        
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.gif':
            imageio.mimsave(save_path, frames, fps=fps)
        else:
            imageio.mimsave(save_path, frames, fps=fps, codec='libx264')
        print(f"Video saved to: {save_path}")
