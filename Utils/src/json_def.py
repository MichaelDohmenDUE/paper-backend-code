# Formales JSON Schema für RL-Algorithmen
# Basierend auf Dozenten-Code, erweitert für modulare RL-Systeme

# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

JSON_SCHEMA = {
    "type": "object",
    "required": ["environment", "networks", "modules"],
    "properties": {
        "environment": {
            "type": "object",
            "required": ["name", "observation_size", "action_size"],
            "properties": {
                "name": {"type": "string"},
                "observation_size": {"type": "integer"},
                "action_size": {"type": "integer"},
                "action_space": {"type": "string", "enum": ["discrete", "continuous"]}
            }
        },

        "networks": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {  # Network names
                    "oneOf": [
                        {
                            # Standard Network Definition
                            "type": "object",
                            "required": ["nodes", "input", "connections"],
                            "properties": {
                                "nodes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["name", "type"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                            # Additional properties for layer parameters
                                        },
                                        "additionalProperties": True
                                    }
                                },
                                "input": {"type": "string"},
                                "connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["from", "to"],
                                        "properties": {
                                            "from": {"type": "string"},
                                            "to": {"type": "string"},
                                            "port": {"type": "string"}  # Optional for multi-input
                                        }
                                    }
                                }
                            }
                        },
                        {
                            # Clone Network Definition
                            "type": "object",
                            "required": ["type", "clone_from"],
                            "properties": {
                                "type": {"const": "clone"},
                                "clone_from": {"type": "string"}
                            }
                        }
                    ]
                }
            }
        },

        "modules": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {  # Module names
                    "oneOf": [
                        {
                            # Graph Module Definition
                            "type": "object",
                            "required": ["nodes"],
                            "properties": {
                                "nodes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["name", "type"],
                                        "additionalProperties": True
                                    }
                                },
                                "input": {"type": "string"},
                                "connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["from", "to"],
                                        "properties": {
                                            "from": {"type": "string"},
                                            "to": {"type": "string"},
                                            "port": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        {
                            # Parameter-only Module (Synchronization)
                            "type": "object",
                            "additionalProperties": True
                        }
                    ]
                }
            }
        }
    }
}

# ============================================================================
# DQN EXAMPLE
# ============================================================================
# TODO: Throw this into utils for now, think about file structure later
DQN_EXAMPLE = {
    "environment": {
        "name": "CartPole-v1",
        "observation_size": 4,
        "action_size": 2,
        "action_space": "discrete"
    },

    "networks": {
        "q_network": {
            "nodes": [
                {"name": "fc1", "type": "Linear", "input_features": 4, "output_features": 64},
                {"name": "relu", "type": "ReLU"},
                {"name": "fc2", "type": "Linear", "input_features": 64, "output_features": 2}
            ],
            "input": "fc1",
            "connections": [
                {"from": "fc1", "to": "relu"},
                {"from": "relu", "to": "fc2"}
            ]
        },

        "q_network_target": {
            "type": "clone",
            "clone_from": "q_network"
        }
    },

    "modules": {
        "data_collection": {
            "nodes": [
                {"name": "state", "type": "StateSource"},
                {"name": "q_net", "type": "NetworkRef", "ref": "q_network"},
                {"name": "epsilon_greedy", "type": "EpsilonGreedy", "epsilon": 0.1},
                {"name": "environment", "type": "Environment"}
            ],
            "input": "state",
            "connections": [
                {"from": "state", "to": "q_net"},
                {"from": "q_net", "to": "epsilon_greedy"},
                {"from": "epsilon_greedy", "to": "environment"}
            ]
        },

        "data_buffering": {
            "nodes": [
                {"name": "replay_buffer", "type": "ReplayBuffer", "capacity": 10000}
            ]
        },

        "learning": {
            "nodes": [
                {"name": "batch", "type": "BatchSource"},
                {"name": "batch_split", "type": "BatchSplitter"},
                {"name": "q_net", "type": "NetworkRef", "ref": "q_network"},
                {"name": "q_target", "type": "NetworkRef", "ref": "q_network_target"},
                {"name": "gather", "type": "Gather"},
                {"name": "qmax", "type": "QMax"},
                {"name": "target_comp", "type": "TargetComputation", "gamma": 0.99},
                {"name": "loss", "type": "MSELoss"},
                {"name": "optimizer", "type": "Adam", "lr": 0.001, "optimizes": "q_network"}
            ],
            "input": "batch",
            "connections": [
                {"from": "batch", "to": "batch_split"},
                {"from": "batch_split", "to": "q_net", "port": "states"},
                {"from": "batch_split", "to": "q_target", "port": "next_states"},
                {"from": "q_net", "to": "gather"},
                {"from": "batch_split", "to": "gather", "port": "actions"},
                {"from": "q_target", "to": "qmax"},
                {"from": "qmax", "to": "target_comp"},
                {"from": "batch_split", "to": "target_comp", "port": "rewards"},
                {"from": "batch_split", "to": "target_comp", "port": "dones"},
                {"from": "gather", "to": "loss"},
                {"from": "target_comp", "to": "loss"}
            ]
        },

        "synchronization": {
            "pairs": [
                {
                    "source": "q_network",
                    "target": "q_network_target",
                    "mode": "hard",
                    "update_frequency": 100,
                    "tau": 1.0
                }
            ],
            "warmup_steps": 0
        }
    }
}

# ============================================================================
# DDPG EXAMPLE
# ============================================================================

DDPG_EXAMPLE = {
    "environment": {
        "name": "Pendulum-v1",
        "observation_size": 3,
        "action_size": 1,
        "action_space": "continuous"
    },

    "networks": {
        "actor": {
            "nodes": [
                {"name": "fc1", "type": "Linear", "input_features": 3, "output_features": 32},
                {"name": "relu", "type": "ReLU"},
                {"name": "fc2", "type": "Linear", "input_features": 32, "output_features": 1},
                {"name": "tanh", "type": "Tanh"},
                {"name": "scale", "type": "Scale", "factor": 2.0}
            ],
            "input": "fc1",
            "connections": [
                {"from": "fc1", "to": "relu"},
                {"from": "relu", "to": "fc2"},
                {"from": "fc2", "to": "tanh"},
                {"from": "tanh", "to": "scale"}
            ]
        },

        "critic": {
            "nodes": [
                {"name": "state_fc", "type": "Linear", "input_features": 3, "output_features": 32},
                {"name": "relu1", "type": "ReLU"},
                {"name": "concat", "type": "Concat", "dim": -1},
                {"name": "fc2", "type": "Linear", "input_features": 33, "output_features": 32},
                {"name": "relu2", "type": "ReLU"},
                {"name": "fc3", "type": "Linear", "input_features": 32, "output_features": 1}
            ],
            "input": "state_fc",
            "connections": [
                {"from": "state_fc", "to": "relu1"},
                {"from": "relu1", "to": "concat"},
                {"from": "concat", "to": "fc2"},
                {"from": "fc2", "to": "relu2"},
                {"from": "relu2", "to": "fc3"}
            ]
        },

        "actor_target": {"type": "clone", "clone_from": "actor"},
        "critic_target": {"type": "clone", "clone_from": "critic"}
    },

    "modules": {
        "data_collection": {
            "nodes": [
                {"name": "state", "type": "StateSource"},
                {"name": "actor", "type": "NetworkRef", "ref": "actor"},
                {"name": "noise", "type": "GaussianNoise", "std": 0.1},
                {"name": "environment", "type": "Environment"}
            ],
            "input": "state",
            "connections": [
                {"from": "state", "to": "actor"},
                {"from": "actor", "to": "noise"},
                {"from": "noise", "to": "environment"}
            ]
        },

        "data_buffering": {
            "nodes": [
                {"name": "replay_buffer", "type": "ReplayBuffer", "capacity": 10000}
            ]
        },

        "learning": {
            "nodes": [
                {"name": "batch", "type": "BatchSource"},
                {"name": "batch_split", "type": "BatchSplitter"},
                {"name": "actor", "type": "NetworkRef", "ref": "actor"},
                {"name": "critic", "type": "NetworkRef", "ref": "critic"},
                {"name": "actor_target", "type": "NetworkRef", "ref": "actor_target"},
                {"name": "critic_target", "type": "NetworkRef", "ref": "critic_target"},
                {"name": "critic_loss", "type": "MSELoss"},
                {"name": "actor_loss", "type": "NegativeMean"},
                {"name": "critic_optimizer", "type": "Adam", "lr": 0.001, "optimizes": "critic"},
                {"name": "actor_optimizer", "type": "Adam", "lr": 0.0001, "optimizes": "actor"}
            ],
            "input": "batch",
            "connections": [
                {"from": "batch", "to": "batch_split"},
                # Critic Loss Path
                {"from": "batch_split", "to": "critic", "port": "states"},
                {"from": "batch_split", "to": "critic", "port": "actions"},
                {"from": "batch_split", "to": "actor_target", "port": "next_states"},
                {"from": "actor_target", "to": "critic_target", "port": "actions"},
                {"from": "batch_split", "to": "critic_target", "port": "next_states"},
                {"from": "critic", "to": "critic_loss", "port": "current_q"},
                {"from": "critic_target", "to": "critic_loss", "port": "target_q"},
                # Actor Loss Path
                {"from": "batch_split", "to": "actor", "port": "states"},
                {"from": "actor", "to": "critic", "port": "predicted_actions"},
                {"from": "critic", "to": "actor_loss"}
            ]
        },

        "synchronization": {
            "pairs": [
                {
                    "source": "actor",
                    "target": "actor_target",
                    "mode": "soft",
                    "tau": 0.005,
                    "update_frequency": 1
                },
                {
                    "source": "critic",
                    "target": "critic_target",
                    "mode": "soft",
                    "tau": 0.005,
                    "update_frequency": 1
                }
            ],
            "warmup_steps": 0
        }
    }
}

# ============================================================================
# PPO EXAMPLE
# ============================================================================

PPO_EXAMPLE = {
    "environment": {
        "name": "CartPole-v1",
        "observation_size": 4,
        "action_size": 2,
        "action_space": "discrete"
    },

    "networks": {
        "actor": {
            "nodes": [
                {"name": "fc1", "type": "Linear", "input_features": 4, "output_features": 16},
                {"name": "relu", "type": "ReLU"},
                {"name": "fc2", "type": "Linear", "input_features": 16, "output_features": 2},
                {"name": "softmax", "type": "Softmax", "dim": 1}
            ],
            "input": "fc1",
            "connections": [
                {"from": "fc1", "to": "relu"},
                {"from": "relu", "to": "fc2"},
                {"from": "fc2", "to": "softmax"}
            ]
        },

        "critic": {
            "nodes": [
                {"name": "fc1", "type": "Linear", "input_features": 4, "output_features": 16},
                {"name": "relu", "type": "ReLU"},
                {"name": "fc2", "type": "Linear", "input_features": 16, "output_features": 1}
            ],
            "input": "fc1",
            "connections": [
                {"from": "fc1", "to": "relu"},
                {"from": "relu", "to": "fc2"}
            ]
        }
    },

    "modules": {
        "data_collection": {
            "nodes": [
                {"name": "state", "type": "StateSource"},
                {"name": "actor", "type": "NetworkRef", "ref": "actor"},
                {"name": "sampler", "type": "MultinomialSampler"},
                {"name": "critic", "type": "NetworkRef", "ref": "critic"},
                {"name": "environment", "type": "Environment"}
            ],
            "input": "state",
            "connections": [
                {"from": "state", "to": "actor"},
                {"from": "state", "to": "critic"},
                {"from": "actor", "to": "sampler"},
                {"from": "sampler", "to": "environment"}
            ]
        },

        "data_buffering": {
            "nodes": [
                {"name": "rollout_buffer", "type": "RolloutBuffer", "size": 1024},
                {"name": "advantages_buffer", "type": "TensorBuffer"},
                {"name": "old_probs_buffer", "type": "TensorBuffer"}
            ]
        },

        "advantage_estimation": {
            "nodes": [
                {"name": "rollout", "type": "RolloutSource"},
                {"name": "values", "type": "ValueSource"},
                {"name": "gae", "type": "GAE", "gamma": 0.99, "lambda": 0.95}
            ],
            "input": "rollout",
            "connections": [
                {"from": "rollout", "to": "gae"},
                {"from": "values", "to": "gae"}
            ]
        },

        "learning": {
            "nodes": [
                {"name": "rollout", "type": "RolloutSource"},
                {"name": "advantages", "type": "AdvantageSource"},
                {"name": "old_probs", "type": "OldProbsSource"},
                {"name": "actor", "type": "NetworkRef", "ref": "actor"},
                {"name": "critic", "type": "NetworkRef", "ref": "critic"},
                {"name": "ratio", "type": "RatioComputation"},
                {"name": "clipped_loss", "type": "ClippedLoss", "epsilon": 0.2},
                {"name": "value_loss", "type": "MSELoss"},
                {"name": "actor_optimizer", "type": "Adam", "lr": 0.0003, "optimizes": "actor"},
                {"name": "critic_optimizer", "type": "Adam", "lr": 0.001, "optimizes": "critic"}
            ],
            "input": "rollout",
            "connections": [
                {"from": "rollout", "to": "actor"},
                {"from": "rollout", "to": "critic"},
                {"from": "actor", "to": "ratio"},
                {"from": "old_probs", "to": "ratio"},
                {"from": "ratio", "to": "clipped_loss"},
                {"from": "advantages", "to": "clipped_loss"},
                {"from": "critic", "to": "value_loss"}
            ]
        }
    }
}

# ============================================================================
# SYNCHRONIZATION SCHEMA DEFINITION
# ============================================================================

SYNCHRONIZATION_SCHEMA = {
    "type": "object",
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "mode"],
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source network name to copy FROM"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target network name to copy TO"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hard", "soft"],
                        "description": "Update type: hard=copy, soft=interpolate"
                    },
                    "tau": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Soft update interpolation factor. Required for mode='soft'"
                    },
                    "update_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Update every N optimizer steps",
                        "default": 1
                    },
                    "start_after_step": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Start updates after N steps",
                        "default": 0
                    }
                },
                "allOf": [
                    {
                        "if": {"properties": {"mode": {"const": "soft"}}},
                        "then": {"required": ["tau"]}
                    }
                ]
            }
        },
        "warmup_steps": {
            "type": "integer",
            "minimum": 0,
            "description": "Global warmup before any synchronization",
            "default": 0
        }
    }
}

# ============================================================================
# SYNCHRONIZATION EXAMPLES
# ============================================================================

SYNC_EXAMPLES = {
    "dqn_hard": {
        "pairs": [
            {
                "source": "q_network",
                "target": "q_network_target",
                "mode": "hard",
                "update_frequency": 100,
                "tau": 1.0  # Optional for hard mode, but can be explicit
            }
        ],
        "warmup_steps": 1000
    },

    "ddpg_soft": {
        "pairs": [
            {
                "source": "actor",
                "target": "actor_target",
                "mode": "soft",
                "tau": 0.005,
                "update_frequency": 1
            },
            {
                "source": "critic",
                "target": "critic_target",
                "mode": "soft",
                "tau": 0.005,
                "update_frequency": 1
            }
        ],
        "warmup_steps": 0
    },

    "mixed_strategy": {
        "pairs": [
            {
                "source": "q_network",
                "target": "q_network_target",
                "mode": "hard",
                "update_frequency": 1000,
                "start_after_step": 5000
            },
            {
                "source": "actor",
                "target": "actor_target",
                "mode": "soft",
                "tau": 0.01,
                "update_frequency": 10
            }
        ],
        "warmup_steps": 1000
    }
}

# ============================================================================
# BACKEND INFERENCE RULES
# ============================================================================

# Implizite Module-Kommunikation Rules für Backend
MODULE_CONTRACTS = {
    "data_collection": {
        "produces": ["transitions", "rollouts", "values", "action_probs"]
    },
    "data_buffering": {
        "consumes": ["transitions", "rollouts"],
        "produces": ["batches", "rollout_data"]
    },
    "advantage_estimation": {
        "consumes": ["rollouts", "values"],
        "produces": ["advantages"]
    },
    "learning": {
        "consumes": ["batches", "rollout_data", "advantages"]
    },
    "synchronization": {
        "consumes": ["network_parameters"],
        "produces": ["synchronized_networks"],
        "trigger": "optimizer_step"
    }
}
