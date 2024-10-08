import pygame


class ManualPolicy:

    def __init__(self, env, agent_id="player_0", blocking=True):

        self.env = env
        self.agent_id = agent_id

        self.blocking = blocking

        # action mappings for all agents are the same

        self.default_action = 0
        self.action_mapping = {pygame.K_SPACE: 0, pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3,
                               pygame.K_RIGHT: 4, pygame.K_f: 5}

    def __call__(self, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent_id
        ), f"Manual Policy only applied to agent: {self.agent_id}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        # if we get a key, override action using the dict
        action_done = False
        while not action_done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # escape to end
                        exit()
                        action_done = True

                    elif event.key == pygame.K_BACKSPACE:
                        # backspace to reset
                        self.env.reset()

                    elif event.key == pygame.K_v:
                        self.env.unwrapped.screenshot()

                    elif event.key in self.action_mapping:
                        action = self.action_mapping[event.key]
                        action_done = True

            if not self.blocking:
                action_done = True

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping

