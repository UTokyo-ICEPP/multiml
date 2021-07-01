from ..basic import ConnectionRandomSearchAgent


class PytorchConnectionRandomSearchAgent(ConnectionRandomSearchAgent):
    """Pytorch implementation for ConnectionRandomSearchAgent."""

    from multiml.task.pytorch import ModelConnectionTask
    _ModelConnectionTask = ModelConnectionTask

    @staticmethod
    def _set_trainable_flags(model, do_trainable):
        # TODO: check
        for params in model.parameters():
            params.require_grad = do_trainable
