from manim import *
import numpy as np
import matplotlib 
import matplotlib.colors as mcolors

# Global colormap object (viridis goes from 0 to 1)
_viridis = matplotlib.colormaps['viridis']

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


# @ run using 
# manim -pql threed_surface.py MatrixConeSurface

# @ name  Elliptic SPD Tube with fixed and varying axes

config.background_color = WHITE  # global white background


class MatrixConeSurface(ThreeDScene):
    def make_cone_surface(self, u_vec, v_vec, z_min=0.0, z_max=1.0,
                          c=1.0, color=RED):
        """
        u_vec, v_vec: 2D orthonormal column vectors (numpy arrays of shape (2,))
        Represent M(z) = exp(z) u u^T + v v^T.
        """

        # Normalize to be safe
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)

        # Sanity check: orthogonality
        if abs(np.dot(u_vec, v_vec)) > 1e-6:
            raise ValueError("u and v must be orthonormal")

        def param_point(theta, z):
            # Semi-axis lengths from eigenvalues of M(z)
            a = c * np.exp(z / 2.0)  # along u (sqrt(exp(z)))
            b = c * np.exp(1)                # along v (fixed, scaled)

            # 2D point in xy-plane
            xy = a * np.cos(theta) * u_vec + b * np.sin(theta) * v_vec

            # Embed in 3D: (x, y, z)
            return np.array([xy[0], xy[1], z])

        surface = Surface(
            lambda theta, zz: param_point(theta, zz),
            u_range=[0, TAU],        # theta ∈ [0, 2π]
            v_range=[z_min, z_max],  # z ∈ [z_min, z_max]
            resolution=(64, 32),
        )

        surface.set_fill(color, opacity=0.6)
        surface.set_stroke(color, width=1)
        surface.set_shade_in_3d(True)

        return surface

    def construct(self):
        # Axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 2, 0.25],
        )
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)
        self.add(axes)

        # --------- Orthonormal basis (u, v) ---------
        u1 = np.array([1.0, 0.0])
        v1 = np.array([0.0, 1.0])

        z_min = 0.0
        z_max = 1.0
        c = 1.0
        color = RED

        # Tracker for current top z
        z_tracker = ValueTracker(z_min)

        # Closure that rebuilds the surface up to current z
        def param_point(theta, alpha):
            """
            alpha ∈ [0, 1] is a normalized vertical parameter.
            We map alpha to z in [z_min, current_top].
            """
            # normalize basis
            uu = u1 / np.linalg.norm(u1)
            vv = v1 / np.linalg.norm(v1)

            current_top = z_tracker.get_value()
            # if current_top == z_min, everything collapses to the bottom ellipse
            z = z_min + alpha * (current_top - z_min)

            a = c * np.exp(z / 2.0)
            b = c * np.exp(1)

            xy = a * np.cos(theta) * uu + b * np.sin(theta) * vv
            return np.array([xy[0], xy[1], z])

        # always_redraw -> rebuilds the surface for each frame
        cone1 = always_redraw(
            lambda: Surface(
                lambda theta, alpha: param_point(theta, alpha),
                u_range=[0, TAU],
                v_range=[0, 1],        # alpha ∈ [0, 1]
                resolution=(64, 32),
            ).set_fill(color, opacity=0.6).set_stroke(color, width=1)
        )

        colorscale = viridis_colorscale(n=int(2/0.25) + 1, z_min=0.0, z_max=2.0)
        cone1.set_style(fill_opacity=0.9, stroke_width=0.5)
        cone1.set_fill_by_value(
            axes=axes,
            colorscale=colorscale,  # list of (color, z_value) tuples
            axis=2,                 # 2 = z-axis
        )


        self.add(cone1)

        # Animate from bottom to top: z_min -> z_max
        self.play(z_tracker.animate.set_value(z_max),
                  run_time=3,
                  rate_func=linear)
        self.wait()