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
            vmax = self.cfg.vmax_epi if label == "Epi" else m.max()
            im = axes[i].imshow(m, cmap=self.unc_cmap, origin='lower', aspect='equal', extent=extent, vmin=0, vmax=vmax)
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

    def render_combined_grid(self, maps_dict, extent, agent_trail=None, ref_traj_world=None,
                             opt_traj_world=None, current_pos=None, current_heading=0.0,
                             step=None, save_path=None, avg_cost=None, current_cost=None,
                             det_conf=None):
        """
        Renders a diagnostic grid of maps with trajectory overlays:
        [RGB, ViewSim]
        [Uncertainty, Occupancy, BEVSim]
        """
        fig = plt.figure(figsize=(16, 10))
        if avg_cost is not None and current_cost is not None:
            fig.suptitle(f"Step {step} | Current Cost: {current_cost:.2f} | Avg Cost: {avg_cost:.2f}", fontsize=16)
        elif step is not None:
            fig.suptitle(f"Step {step}", fontsize=16)

        gs = fig.add_gridspec(2, 3)
        
        # We will use 6 axes
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_sim2d = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_epi = fig.add_subplot(gs[1, 0])
        ax_occ = fig.add_subplot(gs[1, 1])
        ax_sim = fig.add_subplot(gs[1, 2])
        
        min_x, max_x, min_z, max_z = extent
        arrow_len = (max_x - min_x) * 0.05

        def _overlays(ax):
            if ax is None: return
            # Full agent trail
            if agent_trail and len(agent_trail) > 1:
                tx = [p[0] for p in agent_trail]
                tz = [p[1] for p in agent_trail]
                ax.plot(tx, tz, color='gray', linewidth=2, alpha=0.7, zorder=3)

            # A* reference path
            if ref_traj_world and len(ref_traj_world) > 1:
                rx = [p[0] for p in ref_traj_world]
                rz = [p[1] for p in ref_traj_world]
                ax.plot(rx, rz, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)

            # MPPI optimised path
            if opt_traj_world and len(opt_traj_world) > 1:
                ox = [p[0] for p in opt_traj_world]
                oz = [p[1] for p in opt_traj_world]
                ax.plot(ox, oz, color='cyan', linewidth=2, alpha=0.9, zorder=5)
                ax.scatter(ox[-1], oz[-1], marker='*', color='cyan', s=100, zorder=6)

            # Current pose arrow
            if current_pos is not None:
                cx, cz = current_pos
                dx = arrow_len * np.cos(current_heading)
                dz = arrow_len * np.sin(current_heading)
                ax.annotate('', xy=(cx + dx, cz + dz), xytext=(cx, cz),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2.5), zorder=7)
                ax.scatter(cx, cz, color='red', s=30, zorder=7)

        # 0. RGB
        rgb = maps_dict.get('rgb')
        if rgb is not None:
            ax_rgb.imshow(rgb)
            title = f"Agent View (Step {step})" if step is not None else "Agent View"
            ax_rgb.set_title(title)
            ax_rgb.axis('off')
            if det_conf is not None:
                ax_rgb.text(0.5, -0.04, f"Detection confidence: {det_conf:.3f}",
                            transform=ax_rgb.transAxes, ha='center', va='top',
                            fontsize=10)
        
        # 1. View Similarity (2D)
        sim2d = maps_dict.get('sim2d')
        if sim2d is not None:
            s2d = self.normalize_sim(sim2d)
            s2d = self.apply_temperature(s2d, 0.5)
            im1 = ax_sim2d.imshow(s2d, cmap=self.sim_cmap, vmin=0, vmax=1)
            ax_sim2d.set_title("View Similarity")
            plt.colorbar(im1, ax=ax_sim2d, fraction=0.046, pad=0.02, shrink=0.7)
            ax_sim2d.axis('off')

        # 2. Uncertainty
        epi = maps_dict.get('epi')
        if epi is not None:
            im3 = ax_epi.imshow(epi, cmap=self.unc_cmap, origin='lower', extent=extent,vmin=0,vmax=self.cfg.vmax_epi)
            ax_epi.set_title("Epistemic Uncertainty")
            plt.colorbar(im3, ax=ax_epi, fraction=0.046, pad=0.02, shrink=0.7)
            _overlays(ax_epi)

        # 3. Occupancy
        occ = maps_dict.get('occ')
        if occ is not None:
            im4 = ax_occ.imshow(occ, cmap=self.occ_cmap, origin='lower', extent=extent, vmin=0, vmax=2)
            ax_occ.set_title("Occupancy + Plan")
            cbar = plt.colorbar(im4, ax=ax_occ, fraction=0.046, pad=0.02, shrink=0.7)
            cbar.set_ticks([0.33, 1.0, 1.66]); cbar.set_ticklabels(['U', 'F', 'O'])
            _overlays(ax_occ)

        # 4. BEV Similarity
        sim = maps_dict.get('sim')
        if sim is not None:
            s_bev = self.normalize_sim(sim)
            s_bev = self.apply_temperature(s_bev, 0.5)
            im5 = ax_sim.imshow(s_bev, cmap=self.sim_cmap, origin='lower', extent=extent, vmin=0, vmax=1)
            ax_sim.set_title("BEV Similarity")
            plt.colorbar(im5, ax=ax_sim, fraction=0.046, pad=0.02, shrink=0.7)
            _overlays(ax_sim)

        # 5. Overlay Uncertainty over Occupancy
        if epi is not None and occ is not None:
            ax_overlay.imshow(occ, cmap=self.occ_cmap, origin='lower', extent=extent, vmin=0, vmax=2)
            im_overlay = ax_overlay.imshow(epi, cmap=self.unc_cmap, origin='lower', extent=extent, vmin=0, vmax=self.cfg.vmax_epi, alpha=0.6)
            ax_overlay.set_title("Uncertainty + Occupancy")
            plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.02, shrink=0.7)
            _overlays(ax_overlay)

        for ax in [ax_epi, ax_occ, ax_sim, ax_overlay]:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3, linestyle='--')

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

    def render_nav_frame(self, maps_dict, extent, agent_trail, ref_traj_world,
                         opt_traj_world, current_pos, current_heading, step=None):
        """Render a 1×3 navigation frame: occupancy | similarity | uncertainty.

        All trajectory arguments use world-space (x, z) metre coordinates.

        Parameters
        ----------
        maps_dict : dict with keys 'occ', 'sim', 'epi' (numpy arrays)
        extent    : [min_x, max_x, min_z, max_z]
        agent_trail     : list of (x, z) — full position history
        ref_traj_world  : list of (x, z) — current A* reference path
        opt_traj_world  : list of (x, z) — current MPPI optimised path
        current_pos     : (x, z) current agent position
        current_heading : float, radians (atan2(fwd_z, fwd_x) convention)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        min_x, max_x, min_z, max_z = extent
        arrow_len = (max_x - min_x) * 0.08

        def _overlays(ax):
            # Full agent trail
            if len(agent_trail) > 1:
                tx = [p[0] for p in agent_trail]
                tz = [p[1] for p in agent_trail]
                ax.plot(tx, tz, color='gray', linewidth=2.5, alpha=0.90, zorder=3)
                # ax.scatter(tx[0], tz[0], color='lime', s=30, zorder=4, label='Start')

            # A* reference path
            if ref_traj_world and len(ref_traj_world) > 1:
                rx = [p[0] for p in ref_traj_world]
                rz = [p[1] for p in ref_traj_world]
                ax.plot(rx, rz, color='orange', linestyle='--', linewidth=2,
                        alpha=0.85, zorder=4, label='A* ref')

            # MPPI optimised path
            if opt_traj_world and len(opt_traj_world) > 1:
                ox = [p[0] for p in opt_traj_world]
                oz = [p[1] for p in opt_traj_world]
                ax.plot(ox, oz, color='cyan', linewidth=2.5, alpha=0.9, zorder=5, label='MPPI')
                ax.scatter(ox[-1], oz[-1], marker='*', color='cyan', s=200, zorder=6)

            # Current pose arrow
            if current_pos is not None:
                cx, cz = current_pos
                dx = arrow_len * np.cos(current_heading)
                dz = arrow_len * np.sin(current_heading)
                ax.annotate('', xy=(cx + dx, cz + dz), xytext=(cx, cz),
                            arrowprops=dict(arrowstyle='->', color='red', lw=4.0), zorder=7)
                ax.scatter(cx, cz, color='red', s=50, zorder=7)

        title_suffix = f" (step {step})" if step is not None else ""

        # Panel 0 — Occupancy
        occ = maps_dict.get('occ')
        if occ is not None:
            axes[0].imshow(occ, cmap=self.occ_cmap, origin='lower', aspect='equal',
                           extent=extent, vmin=0, vmax=2)
        axes[0].set_title(f"Occupancy + Plan{title_suffix}", fontsize=12)
        _overlays(axes[0])
        axes[0].legend(loc='upper right', fontsize=8, framealpha=0.7)

        # Panel 1 — Similarity
        sim = maps_dict.get('sim')
        if sim is not None:
            sim_vis = self.normalize_sim(sim)
            sim_vis = self.apply_temperature(sim_vis, 0.5)
            im1 = axes[1].imshow(sim_vis, cmap=self.sim_cmap, origin='lower', aspect='equal',
                                 extent=extent, vmin=0, vmax=1)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, shrink=0.7)
        axes[1].set_title(f"Similarity + Trail{title_suffix}", fontsize=12)
        _overlays(axes[1])

        # Panel 2 — Epistemic uncertainty
        epi = maps_dict.get('epi')
        if epi is not None:
            im2 = axes[2].imshow(epi, cmap=self.unc_cmap, origin='lower', aspect='equal',
                                 extent=extent, vmin=0, vmax=self.cfg.vmax_epi)
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02, shrink=0.7)
        axes[2].set_title(f"Uncertainty + MPPI{title_suffix}", fontsize=12)
        _overlays(axes[2])

        for ax in axes:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
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
