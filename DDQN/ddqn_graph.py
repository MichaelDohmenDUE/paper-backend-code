from backend.Utils.src.NodeLib.NodeLibrary import NodeLibrary as NL


def build_ddqn_graph():
    return [
        NL.sample_batch(),
        NL.move_batch_to_device(),
        NL.unpack_batch(),
        NL.compute_q_values(),
        NL.gather_qsa(),
        NL.compute_next_q_online(),
        NL.compute_next_actions(),
        NL.compute_next_q_target(),
        NL.compute_qsa_target(),
        NL.compute_ddqn_target(),
        NL.compute_mse_loss(),
        NL.optimizer_step(),
    ]
