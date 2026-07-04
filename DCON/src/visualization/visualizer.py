import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

    def render_combined_grid(self, maps_dict, extent, agent_trail=None,
                             opt_traj_world=None, current_pos=None, current_heading=0.0,
                             step=None, save_path=None,
                             det_conf=None, goal_world=None, goal_cell=None,
                             mode=None, w_conf=None):
        """
        Renders a diagnostic grid of maps with trajectory overlays:
        [RGB, _, BEVSim]
        [Uncertainty, Occupancy, Uncertainty+Occupancy]
        """
        fig = plt.figure(figsize=(16, 10))
        title_bits = []
        if step is not None:
            title_bits.append(f"Step {step}")
        if mode is not None:
            mode_color = {'EXPLOIT': '#c01818', 'SEARCH': '#1860c0'}.get(mode, '#444')
            title_bits.append(f"Mode: {mode}")
        if w_conf is not None:
            title_bits.append(f"w_conf: {w_conf:.2f}")
        if goal_world is not None:
            title_bits.append(f"goal: ({goal_world[0]:+.2f}, {goal_world[1]:+.2f})m")
        if title_bits:
            fig.suptitle(" | ".join(title_bits), fontsize=14,
                         color=(mode_color if mode is not None else 'black'))

        gs = fig.add_gridspec(2, 3)

        # Top row: RGB on the left, BEV similarity on the right (promoted
        # from the bottom-right). The top-middle cell is intentionally empty
        # (used to hold View Similarity; removed). Bottom row keeps the
        # three BEV maps, with the Uncertainty+Occupancy overlay swapped in
        # at the bottom-right where BEV similarity used to live.
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_sim = fig.add_subplot(gs[0, 2])
        ax_epi = fig.add_subplot(gs[1, 0])
        ax_occ = fig.add_subplot(gs[1, 1])
        ax_overlay = fig.add_subplot(gs[1, 2])

        min_x, max_x, min_z, max_z = extent
        arrow_len = (max_x - min_x) * 0.05

        def _overlays(ax):
            if ax is None: return
            # Full agent trail
            if agent_trail and len(agent_trail) > 1:
                tx = [p[0] for p in agent_trail]
                tz = [p[1] for p in agent_trail]
                ax.plot(tx, tz, color='gray', linewidth=2, alpha=0.7, zorder=3)

            # MPPI optimised path
            if opt_traj_world and len(opt_traj_world) > 1:
                ox = [p[0] for p in opt_traj_world]
                oz = [p[1] for p in opt_traj_world]
                ax.plot(ox, oz, color='cyan', linewidth=2, alpha=0.9, zorder=5)
                ax.scatter(ox[-1], oz[-1], marker='*', color='cyan', s=100, zorder=6)

            # Planner-selected goal cell (argmax of constrained similarity).
            # Distinct from the MPPI rollout endpoint (cyan star).
            if goal_world is not None:
                gx, gz = goal_world
                ax.scatter(gx, gz, marker='X', color='magenta',
                           edgecolors='black', linewidths=1.2, s=160, zorder=8)

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

        # 1. Uncertainty
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

        # 1-metre gridlines so distances are readable at a glance.
        x_ticks = np.arange(np.ceil(min_x), np.floor(max_x) + 1, 1.0)
        z_ticks = np.arange(np.ceil(min_z), np.floor(max_z) + 1, 1.0)
        for ax in [ax_epi, ax_occ, ax_sim, ax_overlay]:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_xticks(x_ticks)
            ax.set_yticks(z_ticks)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def fig_to_numpy(self, fig):
        """Convert a Matplotlib figure to an RGB numpy array."""
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img
