from backend.Utils.src.NodeLib.NodeLibrary import NodeLibrary as NL

def build_dqn_graph():
    return [
        NL.sample_batch(),
        NL.move_batch_to_device(),
        NL.unpack_batch(),
        NL.compute_q_values(),
        NL.gather_qsa(),
        NL.compute_next_q_target(),
        NL.compute_dqn_target(),
        NL.compute_mse_loss(),
        NL.optimizer_step(),
    ]
