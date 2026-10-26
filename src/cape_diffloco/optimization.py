import torch

def state_alignment(sim_next,diff_next,alpha=0.9):
    return sim_next.detach()+alpha*(diff_next-diff_next.detach())

def differentiable_rollout(state,actions,step_fn,sim_states,alpha=0.9):
    """BPTT demonstration. `sim_states` must come from actual forward simulator
    for research use. `step_fn` is a differentiable dynamics transition.
    Do not claim full physics correctness without contact/PD validation.
    """
    trajectory=[]
    for t in range(actions.shape[0]):
        diff_next=step_fn(state,actions[t])
        state=state_alignment(sim_states[t],diff_next,alpha)
        trajectory.append(state)
    return torch.stack(trajectory)
