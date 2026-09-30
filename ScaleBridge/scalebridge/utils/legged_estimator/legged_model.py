import numpy as np
import pinocchio as pin


class LeggedModel:
    """Floating-base rigid-body model used by the estimator.

    Mirrors legged_control2's legged_model::LeggedModel: generalized coordinates follow the pinocchio
    free-flyer convention (q = [pos, quat_xyzw, joints], v = [lin_vel_local, ang_vel_local, joints]),
    frame Jacobians are expressed in LOCAL_WORLD_ALIGNED, and contacts are listed as 3-DoF then 6-DoF.
    """

    def __init__(self, urdf_path, base_name, six_dof_contact_names, three_dof_contact_names=()):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.num_three_dof_contacts = len(three_dof_contact_names)
        self.num_six_dof_contacts = len(six_dof_contact_names)
        self.end_effector_frame_ids = [self._frame_id(name) for name in [*three_dof_contact_names, *six_dof_contact_names]]
        self.base_frame_id = self._frame_id(base_name)

        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.tau = np.zeros(self.num_joints)

    def _frame_id(self, name):
        if not self.model.existFrame(name):
            raise ValueError(f"Frame {name} not found in the estimator model.")
        return self.model.getFrameId(name)

    @property
    def num_joints(self):
        return self.model.nv - 6

    @property
    def num_contacts(self):
        return self.num_three_dof_contacts + self.num_six_dof_contacts

    @property
    def joint_names(self):
        return list(self.model.names[2:])  # skip "universe" and the free-flyer root joint

    def update(self):
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def get_base_rotation(self):
        return pin.Quaternion(self.q[3:7]).toRotationMatrix()

    def get_frame_placement(self, frame_id):
        return self.data.oMf[frame_id]

    def get_mass_matrix(self):
        mass_matrix = pin.crba(self.model, self.data, self.q)
        return np.triu(mass_matrix) + np.triu(mass_matrix, 1).T

    def get_coriolis_matrix(self):
        return pin.computeCoriolisMatrix(self.model, self.data, self.q, self.v).copy()

    def get_generalized_gravity(self):
        return pin.computeGeneralizedGravity(self.model, self.data, self.q).copy()

    def get_selection_matrix(self):
        return np.hstack([np.zeros((self.num_joints, 6)), np.eye(self.num_joints)])

    def get_contact_jacobian(self):
        jacobian = np.zeros((3 * self.num_three_dof_contacts + 6 * self.num_six_dof_contacts, self.model.nv))
        row = 0
        for i, frame_id in enumerate(self.end_effector_frame_ids):
            frame_jacobian = pin.getFrameJacobian(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
            dim = 3 if i < self.num_three_dof_contacts else 6
            jacobian[row:row + dim] = frame_jacobian[:dim]
            row += dim
        return jacobian

    def get_base_jacobian(self):
        return pin.getFrameJacobian(self.model, self.data, self.base_frame_id, pin.LOCAL_WORLD_ALIGNED)
