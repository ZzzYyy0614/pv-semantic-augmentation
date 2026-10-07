"""Task defaults keep method extensions small and explicit."""


class ResearchTask:
    def validate(self, device):
        return None

    def score(self, train, validation):
        return -train["loss"]

    def after_optimizer_step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def epoch_metadata(self, epoch):
        return {}
