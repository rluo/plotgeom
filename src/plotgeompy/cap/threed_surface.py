from manim import *
import numpy as np
import matplotlib 
import matplotlib.colors as mcolors

# Global colormap object (viridis goes from 0 to 1)
_viridis = matplotlib.colormaps['viridis']
 

# @ run using 
# manim -pql threed_surface.py MatrixConeSurface

# @ name  Elliptic SPD Tube with fixed and varying axes

config.background_color = WHITE  # global white background


def viridis_color(t):
    t = max(0.0, min(1.0, float(t)))
    r, g, b, _ = _viridis(t)
    return (r, g, b)

def viridis_colorscale(n=5, z_min=0.0, z_max=1.0):
    """Return a list of (color, value) pairs for set_fill_by_value."""
    ts = np.linspace(0.0, 1.0, n)
    zs = np.linspace(z_min, z_max, n)
    return [
        (rgb_to_color(viridis_color(t)), z)
        for t, z in zip(ts, zs)
    ]


# @ run using:
# manim -pql threed_surface.py MatrixConeSurface

config.background_color = WHITE  # global white background


class MatrixConeSurface(ThreeDScene):
    def construct(self):
        # Axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 2, 0.25],
        )
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)
        self.add(axes)

        # Orthonormal basis (u, v)
        u1 = np.array([1.0, 0.0])
        v1 = np.array([0.0, 1.0])

        z_min = 0.0
        z_max = 1.0
        c = 1.0

        # Tracker for current top z
        z_tracker = ValueTracker(z_min)

        def make_surface():
            """Rebuild the surface up to the current z_tracker value."""

            current_top = z_tracker.get_value()
            # Avoid degenerate case for colorscale
            top = max(current_top, z_min + 1e-6)

            def param(theta, alpha):
                """
                theta ∈ [0, 2π]
                alpha ∈ [0, 1] -> z ∈ [z_min, current_top]
                """
                uu = u1 / np.linalg.norm(u1)
                vv = v1 / np.linalg.norm(v1)

                z = z_min + alpha * (current_top - z_min)

                a = c * np.exp(z / 2.0)
                b = c * np.exp(1)

                xy = a * np.cos(theta) * uu + b * np.sin(theta) * vv
                x, y = xy
                # IMPORTANT: embed in axes coordinates
                return axes.c2p(x, y, z)

            surf = Surface(
                param,
                u_range=[0, TAU],
                v_range=[0, 1],
                resolution=(64, 32),
            )

            surf.set_style(fill_opacity=0.9, stroke_width=0.5)

            # Viridis along z from z_min to current_top
            colorscale = viridis_colorscale(z_min=z_min, z_max=top, n=7)
            surf.set_fill_by_value(
                axes=axes,
                colorscale=colorscale,
                axis=2,  # color by z
            )
            return surf

        # This will rebuild + recolor the surface every frame
        cone1 = always_redraw(make_surface)

        self.add(cone1)

        # Animate from bottom to top: z_min -> z_max
        self.play(
            z_tracker.animate.set_value(z_max),
            run_time=3,
            rate_func=linear,
        )
        self.wait()