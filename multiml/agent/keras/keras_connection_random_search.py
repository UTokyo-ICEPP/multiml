from ..basic import ConnectionRandomSearchAgent


class KerasConnectionRandomSearchAgent(ConnectionRandomSearchAgent):
    """Keras implementation for ConnectionRandomSearchAgent."""

    from multiml.task.keras import ModelConnectionTask
    _ModelConnectionTask = ModelConnectionTask

    @staticmethod
    def _set_trainable_flags(model, do_trainable):
        model.trainable = do_trainable
        # Maybe below lienes are unnecessary...
        for var in model.layers:
            var.trainable = do_trainable
