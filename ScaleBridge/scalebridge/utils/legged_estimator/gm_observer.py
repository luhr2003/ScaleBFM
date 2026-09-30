import numpy as np
from scipy.linalg import cho_factor, cho_solve


class GmObserver:
    """Generalized-momentum observer estimating external joint torques and the resulting contact wrenches.

    Port of legged_control2's legged_estimation::GmObserver. The observer is the discrete first-order
    low-pass form: tau_ext = beta * M v - LPF(beta * M v + C^T v + S^T tau - g), with the low-pass factor
    gamma = 1 / (1 + 2 pi f_c dt) and beta = (1 - gamma) / (gamma dt).
    """

    def __init__(self, legged_model, cutoff_frequency=10.0):
        self.legged_model = legged_model
        self.cutoff_frequency = cutoff_frequency
        self.low_pass_last = None
        self.tau_ext = np.zeros(legged_model.model.nv)
        self.f_ext = np.zeros(0)

    def reset(self):
        self.low_pass_last = None

    def update(self, dt):
        model = self.legged_model
        momentum = model.get_mass_matrix() @ model.v
        coriolis = model.get_coriolis_matrix()
        gravity = model.get_generalized_gravity()
        selection = model.get_selection_matrix()

        gamma = 1.0 / (1.0 + 2.0 * np.pi * self.cutoff_frequency * dt)
        beta = (1.0 - gamma) / gamma / dt
        alpha = beta * momentum + coriolis.T @ model.v + selection.T @ model.tau - gravity

        if self.low_pass_last is None:
            self.low_pass_last = alpha
        else:
            self.low_pass_last = gamma * self.low_pass_last + (1.0 - gamma) * alpha
        self.tau_ext = beta * momentum - self.low_pass_last

    def get_contact_wrenches(self):
        """Solve J^T f = tau_ext for the contact and base wrenches (world-aligned axes, frame origins)."""
        model = self.legged_model
        jacobian = np.vstack([model.get_contact_jacobian(), model.get_base_jacobian()])

        # Damped normal equations: the stacked Jacobian loses rank when a knee reaches full extension.
        normal = jacobian @ jacobian.T
        normal[np.diag_indices_from(normal)] += 1e-8 * np.max(np.diag(normal))
        self.f_ext = cho_solve(cho_factor(normal), jacobian @ self.tau_ext)

        num_three_dof = model.num_three_dof_contacts
        wrenches = [self.f_ext[3 * i:3 * i + 3].copy() for i in range(num_three_dof)]
        wrenches += [self.f_ext[3 * num_three_dof + 6 * i:3 * num_three_dof + 6 * i + 6].copy() for i in range(model.num_six_dof_contacts)]
        return wrenches

    def get_base_wrench(self):
        return self.f_ext[-6:]
