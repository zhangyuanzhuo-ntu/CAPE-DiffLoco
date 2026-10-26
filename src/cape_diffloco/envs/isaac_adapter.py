class IsaacQuadrupedAdapter:
    """Adapter skeleton for Isaac Gym / Isaac Lab.

    Implement this class once GPU resources and the exact simulator stack are available.
    The rest of the repository should call only this adapter, not simulator-specific APIs.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        raise NotImplementedError(
            "Connect this adapter to Isaac Gym / Isaac Lab. Required methods: reset, get_obs, step, get_state, get_contacts."
        )

    def reset(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_state(self):
        """Return full privileged simulator state for alignment and loss."""
        raise NotImplementedError

    def get_contacts(self):
        """Return contact labels, foot forces, foot positions, slip indicators."""
        raise NotImplementedError
